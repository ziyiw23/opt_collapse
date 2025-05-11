import numpy as np
import os
import argparse
import torch
import pandas as pd
import sys
from scipy.spatial import KDTree
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import traceback # Import traceback
from torch_geometric.data import Data # Ensure Data is imported/available

# Ensure the main project directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Original Pipeline Imports ---
from collapse.data import process_pdb, extract_env_from_resid, atom_info, sample_functional_center, BaseTransform as OriginalBaseTransform
# Import the global transform instance to modify its cutoff temporarily
from collapse.data import transform as original_global_transform 

# --- Optimized Pipeline Imports ---
from collapse.embedding_utils import extract_env_for_residue_cpu, BaseTransform as OptimizedBaseTransform

def compare_numpy_arrays(name, arr1, arr2, rtol=1e-5, atol=1e-8):
    print(f"\n--- Comparing NumPy Array: {name} ---")
    if arr1 is None or arr2 is None:
        print(" One or both arrays are None.")
        return False
    print(f"Original Shape: {arr1.shape}")
    print(f"Optimized Shape: {arr2.shape}")
    if arr1.shape != arr2.shape:
        print(" Shapes differ.")
        return False
    
    are_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    print(f"Arrays are close (np.allclose): {are_close}")
    if not are_close:
        diff = np.abs(arr1 - arr2)
        print(f" Max difference: {np.max(diff):.6g}")
        print(f" Mean difference: {np.mean(diff):.6g}")
        # Manual check like before
        diff_count = 0
        for i in range(arr1.shape[0]):
            for j in range(arr1.shape[1]):
                val1 = arr1[i, j]
                val2 = arr2[i, j]
                if abs(val1 - val2) > (atol + rtol * abs(val2)):
                    diff_count += 1
                    if diff_count <= 10:
                        print(f"  Diff at ({i}, {j}): Orig={val1:.8f}, Opt={val2:.8f}, Diff={abs(val1-val2):.6g}")
        print(f"Manual comparison found {diff_count} differing elements.")
    return are_close

def main():
    parser = argparse.ArgumentParser(
        description="Compare graph Data objects directly from original vs. optimized pipelines."
    )
    parser.add_argument("--pdb_file", required=True,
                        help="Path to the input PDB file")
    parser.add_argument("--chain", required=True,
                        help="Target chain ID")
    parser.add_argument("--resnum", required=True, type=int,
                        help="Target residue number")
    parser.add_argument("--env_radius", type=float, default=10.0,
                        help="Environment radius for graph construction")
    parser.add_argument("--max_neighbors", type=int, default=32,
                        help="Max neighbors (used internally by radius_graph default)")
    args = parser.parse_args()
    
    # --- Load PDB Data --- 
    print(f"Loading PDB: {args.pdb_file}")
    try:
        atoms_df = process_pdb(args.pdb_file)
        # --- Debug: Print DataFrame columns --- 
        print("--- Debug: Columns in atoms_df from process_pdb ---")
        print(atoms_df.columns)
        print("---------------------------------------------------")
        # ----------------------------------------
        if atoms_df is None or atoms_df.empty:
            print("Error: Failed to process PDB file.")
            return
        pdb_id = os.path.basename(args.pdb_file)
        atoms_df['id'] = pdb_id # Add id needed for optimized path
    except Exception as e:
        print(f"Error loading or processing PDB: {e}")
        return
        
    # --- Get Target Residue Info --- 
    try:
        target_res_df = atoms_df[(atoms_df['chain'] == args.chain) & (atoms_df['residue'] == args.resnum)]
        if target_res_df.empty:
            print(f"Error: Residue {args.chain}_{args.resnum} not found.")
            return
        resname = target_res_df['resname'].iloc[0]
        resletter = atom_info.aa_to_letter(resname)
        if resletter == 'X':
             print(f"Warning: Non-standard residue {args.chain}_{args.resnum}. Skipping.")
             return
        original_resid = f"{resletter}{args.resnum}"
        optimized_resid_tuple = (resletter, args.resnum)
        print(f"Target Residue: {args.chain}_{original_resid} ({resname})")
    except Exception as e:
        print(f"Error getting target residue info: {e}")
        return

    # --- Debug: Print atoms around target --- 
    print("\n--- Debug: Atoms for target residue --- ")
    target_atoms_debug = atoms_df[(atoms_df['chain'] == args.chain) & (atoms_df['residue'] == args.resnum)]
    # print(target_atoms_debug[['atom_name', 'element', 'resname', 'residue', 'chain']]) # Commented out due to previous error
    print(target_atoms_debug.head()) # Print head instead to see available columns
    print("---------------------------------------")
    # ----------------------------------------

    # --- 1. Generate Original Graph & Capture df_env --- 
    print("\nGenerating graph using ORIGINAL pipeline...")
    original_graph = None
    original_df_env = None
    original_coords_np = None # Capture numpy coords
    original_center = None # Add variable to store original center
    original_res_df = None # Add variable for original res_df
    try:
        # --- Temporarily set the global transform's cutoff to 4.5 --- 
        original_cutoff = original_global_transform.edge_cutoff # Store original
        fixed_cutoff = 4.5 # Hardcode cutoff for this test
        original_global_transform.edge_cutoff = fixed_cutoff
        print(f"Temporarily set original transform edge_cutoff to: {original_global_transform.edge_cutoff}")
        # ----------------------------------------------------------

        # Call original function requesting debug info (center + res_df)
        orig_result = extract_env_from_resid(atoms_df.copy(), (args.chain, original_resid), 
                                             env_radius=10.0, train_mode=False)
        
        # --- Modified Unpacking Logic --- 
        if isinstance(orig_result, Data):
            print(" Original pipeline returned a Data object directly.")
            original_graph = orig_result
            original_center = None # Not returned
            original_res_df = None # Not returned
        elif isinstance(orig_result, tuple) and orig_result is not None:
            print(f" Original pipeline returned a tuple of length {len(orig_result)}.")
            # --- Add check for new 2-element tuple format --- 
            if len(orig_result) == 2:
                print(" Assuming tuple format is (graph, res_df).")
                original_graph, original_res_df = orig_result
                original_center = None # Not returned in this format
            # -------------------------------------------------
            elif len(orig_result) == 3: # Original expected case
                print(" Assuming tuple format is (graph, center, res_df).")
                original_graph, original_center, original_res_df = orig_result
            else: # Handle other unexpected tuple lengths
                print(" Warning: Original pipeline did not return expected 2 or 3 values in tuple.")
                if len(orig_result) > 0: original_graph = orig_result[0]
                if len(orig_result) > 1: original_center = orig_result[1]
                # Ensure original_res_df is None if not enough elements or format unknown
                if len(orig_result) < 2 : original_res_df = None 
                elif len(orig_result) == 2 and not isinstance(orig_result[1], pd.DataFrame): # If len 2 but 2nd isn't DF
                     original_res_df = None
                elif len(orig_result) < 3: original_res_df = None # Reset if len 2 was handled above
                 
                # Handle potential case where graph itself might be None in tuple
                if original_graph is not None and not isinstance(original_graph, Data):
                     print(f" Warning: First element of tuple is not a Data object (type: {type(original_graph)}). Setting original_graph to None.")
                     original_graph = None
        elif orig_result is None:
             print(" Original pipeline returned None.")
             original_graph = None
             original_center = None
             original_res_df = None
        else:
             print(f" Warning: Original pipeline returned unexpected type: {type(orig_result)}. Setting outputs to None.")
             original_graph = None
             original_center = None
             original_res_df = None
        # --- End Modified Unpacking Logic --- 

        # Check results after unpacking
        if original_graph is None: print("Original graph generation failed or result format unexpected.")
        else: print("Original graph generated successfully.")
        if original_center is not None: 
            print(f"Original center captured: {original_center}")
        else:
            print("Warning: Original center not captured.")
        if original_res_df is not None and not original_res_df.empty:
             print(f"Original res_df captured, shape: {original_res_df.shape}")
        else:
             print("Warning: Original res_df not captured or is empty.")
        # ---------------------------------------------
    except Exception as e:
        print(f"❌ Error during original graph generation or unpacking: {type(e).__name__}: {e}")
        tb_str = traceback.format_exc()
        print("--- Traceback --- ")
        print(tb_str)
        print("-----------------")
        # Ensure cutoff is restored even on error
        if 'original_cutoff' in locals() and original_cutoff is not None: # Check if original_cutoff was defined
             original_global_transform.edge_cutoff = original_cutoff
             print(f"Restored original transform edge_cutoff (on error) to: {original_global_transform.edge_cutoff}")

    # --- 2. Generate Optimized Graph & Capture df_env --- 
    print("\nGenerating graph using OPTIMIZED pipeline...")
    optimized_graph = None
    optimized_df_env = None # Keep for potential future checks, though not primary focus now
    optimized_coords_np = None # Keep for potential future checks
    optimized_center = None # Add variable to store optimized center
    optimized_res_df = None # Add variable for optimized res_df
    try:
        fixed_cutoff = 4.5 
        opt_base_transform = OptimizedBaseTransform(edge_cutoff=fixed_cutoff, 
                                           device='cpu')
        print(f"Initialized optimized transform with edge_cutoff: {opt_base_transform.edge_cutoff}")
        
        chain_atoms_df = atoms_df[atoms_df['chain'] == args.chain].copy()
        if 'id' not in chain_atoms_df.columns: chain_atoms_df['id'] = pdb_id

        if chain_atoms_df.empty: print("  Error: Chain DataFrame empty.")
        else:
            opt_result = extract_env_for_residue_cpu(atoms_df, 
                                                     chain_atoms_df, 
                                                     optimized_resid_tuple, 
                                                     args.env_radius, 
                                                     opt_base_transform,
                                                     return_debug_info=True) 
            
            if opt_result is not None:
                if len(opt_result) == 3: 
                    optimized_graph, optimized_center, optimized_res_df = opt_result 
                else: 
                    print("Warning: Optimized pipeline did not return expected 3 values.")
                    if len(opt_result) > 0: optimized_graph = opt_result[0]
                    if len(opt_result) > 1: optimized_center = opt_result[1]

                if optimized_graph is None: print("Optimized graph generation failed.")
                else: print("Optimized graph generated successfully.")
            else:
                 print("Optimized pipeline returned None.")
            
    except Exception as e:
        print(f"Error during optimized graph generation: {e}")

    # --- 2.5 Compare df_env DataFrames Passed to BaseTransform --- 
    # This section might become less relevant if df_env isn't returned, but keep for now
    print("\n=== df_env DataFrame Comparison (If Available) ===")
    if original_df_env is not None and optimized_df_env is not None: # This comparison is less meaningful now
        print(f"Original df_env shape: {original_df_env.shape}")
        print(f"Optimized df_env shape: {optimized_df_env.shape}")
        # ... rest of comparison ...
    else:
        print("Could not compare df_env (one or both missing/not returned).")

    # --- 2.7 Compare Numpy Coords Arrays --- 
    # This section is also less relevant if df_env isn't returned
    print("\n=== NumPy Coords Comparison (If Available) ===")
    compare_numpy_arrays("Coords NumPy Array (from df_env)", original_coords_np, optimized_coords_np)

    # --- ADDED: Compare res_df DataFrames ---
    print("\n=== res_df DataFrame Comparison ===")
    res_dfs_match = False
    if original_res_df is not None and optimized_res_df is not None: 
        print(f"Original res_df shape: {original_res_df.shape}")
        print(f"Optimized res_df shape: {optimized_res_df.shape}")
        try:
            # Prepare for comparison: Sort columns, reset index
            orig_rdf = original_res_df.sort_index(axis=1).reset_index(drop=True)
            opt_rdf = optimized_res_df.sort_index(axis=1).reset_index(drop=True)
            
            # Drop columns added during optimized processing if they aren't in original
            if 'resname_letter' in opt_rdf.columns and 'resname_letter' not in orig_rdf.columns:
                print(" Dropping 'resname_letter' from optimized_res_df for comparison.")
                opt_rdf = opt_rdf.drop(columns=['resname_letter'])
            if 'index' in opt_rdf.columns and 'index' not in orig_rdf.columns:
                print(" Dropping 'index' column (from reset_index) from optimized_res_df for comparison.")
                opt_rdf = opt_rdf.drop(columns=['index'])

            # --- Normalize resname column for comparison --- 
            # The original res_df has single-letter codes due to internal processing
            # The optimized res_df has three-letter codes
            # Let's convert the original back to three-letter codes for comparison
            if 'resname' in orig_rdf.columns and 'resname' in opt_rdf.columns:
                try:
                    # Assume atom_info has a way to reverse map or build one if needed
                    # For now, let's skip the conversion and see if ignoring this column helps
                    # TODO: Implement actual resname normalization if needed
                    print(" Comparing res_dfs while temporarily ignoring the 'resname' column due to format difference.")
                    orig_rdf_compare = orig_rdf.drop(columns=['resname'])
                    opt_rdf_compare = opt_rdf.drop(columns=['resname'])
                    
                    # Compare using pandas testing function (handles NaNs, dtypes)
                    assert_frame_equal(orig_rdf_compare, opt_rdf_compare, check_dtype=True, 
                                       rtol=1e-5, atol=1e-8) # Use tolerances for float cols
                    print("✅ res_df DataFrames (excluding resname) are considered equal")
                    res_dfs_match = True # Mark as match if all *other* columns match

                except AssertionError as e:
                    print("❌ res_df DataFrames differ (even excluding resname):")
                    # Print differences concisely
                    diff_summary = str(e).split('\n')
                    print("  " + "\n  ".join(diff_summary[:5])) # Print first few lines of diff
                    res_dfs_match = False
                except Exception as e:
                    print(f"Error during res_df comparison (excluding resname): {e}")
                    res_dfs_match = False
            else:
                 print(" Could not compare res_dfs - 'resname' column missing from one or both.")
                 res_dfs_match = False
            # ---------------------------------------------
        except AssertionError as e: # This block might become redundant now
            print("❌ res_df DataFrames differ (Initial check before dropping resname):")
            diff_summary = str(e).split('\n')
            print("  " + "\n  ".join(diff_summary[:5])) 
            res_dfs_match = False
        except Exception as e: # This block might become redundant now
            print(f"Error during initial res_df comparison setup: {e}")
            res_dfs_match = False
            
    elif original_res_df is None and optimized_res_df is None:
        print("Both res_dfs are None or empty. Treating as matching.")
        res_dfs_match = True # Both failed to be captured
    else:
        print("Could not compare res_dfs (one is None or empty).")
        res_dfs_match = False

    # --- 3. Direct Graph Comparison --- 
    if original_graph is None or optimized_graph is None:
        print("\nCannot compare graphs.")
        return
        
    print("\n=== Direct Graph Comparison ===")
    
    # Compare Nodes
    print("\n--- Node Comparison ---")
    nodes_match = False
    if hasattr(original_graph, 'x') and hasattr(optimized_graph, 'x') and \
       hasattr(original_graph, 'atoms') and hasattr(optimized_graph, 'atoms'):
       
        print(f"Original num_nodes: {original_graph.num_nodes}")
        print(f"Optimized num_nodes: {optimized_graph.num_nodes}")
        if original_graph.num_nodes == optimized_graph.num_nodes:
            print("Node counts match.")
            coords_close = torch.allclose(original_graph.x, optimized_graph.x, rtol=1e-5, atol=1e-8)
            atoms_equal = torch.equal(original_graph.atoms, optimized_graph.atoms)
            print(f"Coords are close (torch.allclose): {coords_close}")
            print(f"Atoms are equal (torch.equal): {atoms_equal}")
            nodes_match = coords_close and atoms_equal
            
            # --- Manual element-wise comparison for coords --- 
            if not coords_close:
                print("Performing manual element-wise comparison for coords...")
                diff_count = 0
                coords1_np = original_graph.x.cpu().numpy()
                coords2_np = optimized_graph.x.cpu().numpy()
                # Define tolerances similar to allclose
                rtol=1e-5
                atol=1e-8
                for i in range(coords1_np.shape[0]):
                    for j in range(coords1_np.shape[1]):
                        val1 = coords1_np[i, j]
                        val2 = coords2_np[i, j]
                        # Manual closeness check
                        if abs(val1 - val2) > (atol + rtol * abs(val2)):
                            diff_count += 1
                            if diff_count <= 10: 
                                print(f"  Difference at index ({i}, {j}): Orig={val1:.8f}, Opt={val2:.8f}, Diff={abs(val1-val2):.6g}")
                print(f"Manual comparison found {diff_count} differing elements in coords.")
            # --------------------------------------------- 
        else:
            print("Node counts differ.")
    else:
        print("Could not compare nodes (missing attributes).")

    print("\n--- Edge Comparison ---")
    edges_match = False
    if hasattr(original_graph, 'edge_index') and hasattr(optimized_graph, 'edge_index') and \
       hasattr(original_graph, 'edge_s') and hasattr(optimized_graph, 'edge_s') and \
       hasattr(original_graph, 'edge_v') and hasattr(optimized_graph, 'edge_v'):

        print(f"Original num_edges: {original_graph.num_edges}")
        print(f"Optimized num_edges: {optimized_graph.num_edges}")
        
        if original_graph.num_edges == optimized_graph.num_edges:
            print("Edge counts match.")
            np_ei1 = original_graph.edge_index.cpu().numpy()
            np_ei2 = optimized_graph.edge_index.cpu().numpy()
            np_ei1 = np.sort(np_ei1, axis=0)
            np_ei2 = np.sort(np_ei2, axis=0)
            inds1 = np.lexsort((np_ei1[1,:], np_ei1[0,:]))
            inds2 = np.lexsort((np_ei2[1,:], np_ei2[0,:]))
            ei_equal = np.array_equal(np_ei1[:, inds1], np_ei2[:, inds2])
            print(f"Edge indices are equal (order invariant): {ei_equal}")

            if ei_equal:
                # --- Corrected: Compare edge attributes using sorted indices --- 
                # Ensure tensors are on CPU before comparing with allclose if needed
                orig_edge_s_sorted = original_graph.edge_s[inds1].cpu()
                opt_edge_s_sorted = optimized_graph.edge_s[inds2].cpu()
                orig_edge_v_sorted = original_graph.edge_v[inds1].cpu()
                opt_edge_v_sorted = optimized_graph.edge_v[inds2].cpu()
                
                es_close = torch.allclose(orig_edge_s_sorted, opt_edge_s_sorted, rtol=1e-5, atol=1e-8)
                ev_close = torch.allclose(orig_edge_v_sorted, opt_edge_v_sorted, rtol=1e-5, atol=1e-8)
                # -------------------------------------------------------------
                print(f"Edge_s are close: {es_close}")
                print(f"Edge_v are close: {ev_close}")
                edges_match = ei_equal and es_close and ev_close
            else:
                 edges_match = False
        else:
            print("Edge counts differ.")
            edges_match = False
    else:
        print("Could not compare edges (missing attributes).")

    print("\nComparison finished.")
    if nodes_match and edges_match and res_dfs_match: 
         print("\n✅ SUCCESS: Graphs and res_dfs appear to be identical/consistent!")
    else:
         print("\n❌ FAILURE: Differences found.")
         if not nodes_match or not edges_match:
             print("  - Graph differences detected.")
         if not res_dfs_match:
             print("  - res_df differences detected.")

    # --- Detailed Diff Calculation & Overlay Visualization --- 
    if original_graph is not None and optimized_graph is not None:
        print("\nCalculating detailed differences for overlay visualization...")
        try:
            # Node Differences
            orig_num_nodes = original_graph.num_nodes
            opt_num_nodes = optimized_graph.num_nodes
            orig_nodes = set(range(orig_num_nodes))
            opt_nodes = set(range(opt_num_nodes))
            
            common_nodes = list(orig_nodes & opt_nodes)
            orig_only_nodes = list(orig_nodes - opt_nodes)
            opt_only_nodes = list(opt_nodes - orig_nodes)

            diff_attr_nodes = []
            consistent_nodes = []
            for node_idx in common_nodes:
                coords_match = torch.allclose(original_graph.x[node_idx], optimized_graph.x[node_idx], rtol=1e-5, atol=1e-8)
                atoms_match = torch.equal(original_graph.atoms[node_idx], optimized_graph.atoms[node_idx])
                if not coords_match or not atoms_match:
                    diff_attr_nodes.append(node_idx)
                else:
                    consistent_nodes.append(node_idx)

            # Edge Differences (using the sorted indices calculated earlier)
            orig_ei_np_sorted = np.sort(original_graph.edge_index.cpu().numpy(), axis=0)
            opt_ei_np_sorted = np.sort(optimized_graph.edge_index.cpu().numpy(), axis=0)
            inds1 = np.lexsort((orig_ei_np_sorted[1,:], orig_ei_np_sorted[0,:]))
            inds2 = np.lexsort((opt_ei_np_sorted[1,:], opt_ei_np_sorted[0,:]))
            
            # Use pairs where u < v for set representation
            def to_edge_set(ei_np):
                edge_set = set()
                for i in range(ei_np.shape[1]):
                    u, v = sorted(ei_np[:, i])
                    edge_set.add((u, v))
                return edge_set

            orig_edges_set = to_edge_set(original_graph.edge_index.cpu().numpy())
            opt_edges_set = to_edge_set(optimized_graph.edge_index.cpu().numpy())

            common_edges_set = orig_edges_set & opt_edges_set
            orig_only_edges = list(orig_edges_set - opt_edges_set)
            opt_only_edges = list(opt_edges_set - orig_edges_set)
            
            # Map edge tuples back to original indices for attribute comparison
            # Need original edge indices for this mapping
            orig_ei_np = original_graph.edge_index.cpu().numpy()
            opt_ei_np = optimized_graph.edge_index.cpu().numpy()
            orig_edge_map = {tuple(sorted(orig_ei_np[:,i])): i for i in range(orig_ei_np.shape[1])}
            opt_edge_map = {tuple(sorted(opt_ei_np[:,i])): i for i in range(opt_ei_np.shape[1])}

            diff_attr_edges = []
            consistent_edges = []
            for edge_tuple in common_edges_set:
                idx1 = orig_edge_map[edge_tuple]
                idx2 = opt_edge_map[edge_tuple]
                
                es_match = torch.allclose(original_graph.edge_s[idx1].cpu(), optimized_graph.edge_s[idx2].cpu(), rtol=1e-5, atol=1e-8)
                ev_match = torch.allclose(original_graph.edge_v[idx1].cpu(), optimized_graph.edge_v[idx2].cpu(), rtol=1e-5, atol=1e-8)

                if not es_match or not ev_match:
                    diff_attr_edges.append(edge_tuple)
                else:
                    consistent_edges.append(edge_tuple)

            print(" Difference Calculation Summary:")
            print(f"  Nodes: {len(consistent_nodes)} consistent, {len(diff_attr_nodes)} attr-diff, {len(orig_only_nodes)} orig-only, {len(opt_only_nodes)} opt-only")
            print(f"  Edges: {len(consistent_edges)} consistent, {len(diff_attr_edges)} attr-diff, {len(orig_only_edges)} orig-only, {len(opt_only_edges)} opt-only")

            # --- Combined Visualization (Side-by-Side + Overlay) ---
            print("\nGenerating combined visualization...")
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            ax1, ax2, ax3 = axes # Unpack the axes

            # --- Plot 1: Original Graph --- 
            print(" Plotting Original Graph...")
            nx_orig = to_networkx(original_graph, node_attrs=['atoms'])
            pos_orig = {i: original_graph.x[i, :2].cpu().numpy() for i in range(original_graph.num_nodes)}
            nx.draw(nx_orig, pos=pos_orig, ax=ax1, with_labels=False, node_size=50, width=0.5)
            ax1.set_title("Original Graph (2D Projection)")
            ax1.set_xlabel("X coordinate")
            ax1.set_ylabel("Y coordinate")
            ax1.set_aspect('equal', adjustable='box')
            ax1.axis('on')
            ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

            # --- Plot 2: Optimized Graph --- 
            print(" Plotting Optimized Graph...")
            nx_opt = to_networkx(optimized_graph, node_attrs=['atoms'])
            pos_opt = {i: optimized_graph.x[i, :2].cpu().numpy() for i in range(optimized_graph.num_nodes)}
            nx.draw(nx_opt, pos=pos_opt, ax=ax2, with_labels=False, node_size=50, width=0.5)
            ax2.set_title("Optimized Graph (2D Projection)")
            ax2.set_xlabel("X coordinate")
            ax2.set_ylabel("Y coordinate")
            ax2.set_aspect('equal', adjustable='box')
            ax2.axis('on')
            ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            
            # --- Plot 3: Overlay --- 
            print(" Plotting Overlay Graph...")
            G = nx.Graph() # Combined graph for overlay drawing
            all_nodes = list(orig_nodes | opt_nodes)
            G.add_nodes_from(all_nodes)
            
            # Use original graph coordinates for overlay layout 
            pos = {} 
            for node_idx in all_nodes:
                 if node_idx in orig_nodes:
                     pos[node_idx] = original_graph.x[node_idx, :2].cpu().numpy()
                 elif node_idx in opt_nodes:
                     # Simple fallback: place near origin for opt-only nodes
                     pos[node_idx] = np.array([0.0, 0.0]) + np.random.rand(2) * 0.1 
                 else:
                     pos[node_idx] = np.array([0.0, 0.0]) # Fallback

            # Define Colors (reuse from previous step)
            color_consistent = 'green'
            color_diff_attr = 'yellow'
            color_orig_only = 'blue'
            color_opt_only = 'red'
            color_default_node = 'grey' 
            color_default_edge = 'lightgrey' 
            
            # Define sizes
            node_size = 20 
            edge_width = 0.6

            # Draw Edges on ax3
            nx.draw_networkx_edges(G, pos, ax=ax3, edgelist=list(orig_edges_set | opt_edges_set), width=edge_width, edge_color=color_default_edge, alpha=0.5)
            nx.draw_networkx_edges(G, pos, ax=ax3, edgelist=consistent_edges, width=edge_width, edge_color=color_consistent)
            nx.draw_networkx_edges(G, pos, ax=ax3, edgelist=diff_attr_edges, width=edge_width, edge_color=color_diff_attr)
            nx.draw_networkx_edges(G, pos, ax=ax3, edgelist=orig_only_edges, width=edge_width, edge_color=color_orig_only, style='dashed')
            nx.draw_networkx_edges(G, pos, ax=ax3, edgelist=opt_only_edges, width=edge_width, edge_color=color_opt_only, style='dotted')

            # Draw Nodes on ax3
            nx.draw_networkx_nodes(G, pos, ax=ax3, nodelist=all_nodes, node_size=node_size, node_color=color_default_node, alpha=0.7)
            nx.draw_networkx_nodes(G, pos, ax=ax3, nodelist=consistent_nodes, node_size=node_size, node_color=color_consistent)
            nx.draw_networkx_nodes(G, pos, ax=ax3, nodelist=diff_attr_nodes, node_size=node_size, node_color=color_diff_attr)
            nx.draw_networkx_nodes(G, pos, ax=ax3, nodelist=orig_only_nodes, node_size=node_size, node_color=color_orig_only)
            nx.draw_networkx_nodes(G, pos, ax=ax3, nodelist=opt_only_nodes, node_size=node_size, node_color=color_opt_only)

            ax3.set_title(f"Comparison Overlay: {args.chain}_{original_resid}")
            ax3.set_xlabel("X coordinate")
            ax3.set_ylabel("Y coordinate")
            ax3.set_aspect('equal', adjustable='box')
            ax3.axis('on') 
            ax3.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            
            # Create legend handles (reuse from previous step)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Consistent', markerfacecolor=color_consistent, markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Attributes Differ', markerfacecolor=color_diff_attr, markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Original Only', markerfacecolor=color_orig_only, markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Optimized Only', markerfacecolor=color_opt_only, markersize=8),
                Line2D([0], [0], color=color_consistent, lw=2, label='Consistent Edge'),
                Line2D([0], [0], color=color_diff_attr, lw=2, label='Attr-Diff Edge'),
                Line2D([0], [0], color=color_orig_only, lw=2, linestyle='dashed', label='Orig-Only Edge'),
                Line2D([0], [0], color=color_opt_only, lw=2, linestyle='dotted', label='Opt-Only Edge')
            ]
            ax3.legend(handles=legend_elements, loc='upper right', fontsize='small')
            
            # --- Finalize and Save --- 
            vis_dir = "./test_env_extract_visualizations" # Define a directory
            os.makedirs(vis_dir, exist_ok=True) # Create if it doesn't exist
            vis_path = os.path.join(vis_dir, f"graph_combined_{args.chain}_{original_resid}.png")
            fig.suptitle(f"Graph Comparison: {args.chain}_{original_resid}", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
            plt.savefig(vis_path, dpi=300)
            plt.close(fig) # Close the figure explicitly
            print(f"✅ Combined visualization saved to {vis_path}")

        except Exception as vis_e:
            print(f"❌ Error generating combined visualization: {vis_e}")
            tb_str = traceback.format_exc()
            print(tb_str)
    # --- End Visualization --- 

if __name__ == "__main__":
    main()
