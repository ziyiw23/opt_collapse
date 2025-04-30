# debug_inference_diff.py

import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from torch_geometric.data import Batch, Data
import argparse
import time
from contextlib import nullcontext
import collections as col
import glob

# --- Seeding and Determinism ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA determinism enabled.")
# -----------------------------

# Ensure the main project directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Original Pipeline Imports ---
from collapse.data import process_pdb, atom_info
from collapse.data import extract_env_from_resid as extract_env_from_resid_original
from collapse.data import BaseTransform as OriginalBaseTransform
from collapse import initialize_model # Restored model import

# --- Optimized Pipeline Imports ---
from embedding_utils import GraphPreparationTransformCPU
from atom3d.filters.filters import first_model_filter

# --- Configuration ---
PDB_DIR = "/scratch/groups/rbaltman/ziyiw23/1000_pdbs/"
NUM_PDBS_TO_TEST = 9
CHECKPOINT_PATH = 'data/checkpoints/collapse_base.pt' # Restored checkpoint path
ENV_RADIUS = 10.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
INCLUDE_HETS = False
ORIGINAL_EDGE_CUTOFF = 4.5

# --- Helper: Detailed Graph Object Comparison ---
# ... (compare_graph_objects function remains the same) ...
def compare_graph_objects(g1, g2, resid_label):
    """Compares two torch_geometric Data objects attribute by attribute."""
    match = True
    tensors_to_check = ['x', 'atoms', 'edge_index', 'edge_s', 'edge_v']
    # print(f"   Comparing graph for {resid_label}:") # Make less verbose

    # Quick initial checks
    if g1 is None and g2 is None: return True
    if g1 is None or g2 is None:
         print(f"    ❌ FAIL: Graph is None for {resid_label} in one pathway.")
         return False
    if not isinstance(g1, Data) or not isinstance(g2, Data):
         print(f"    ❌ FAIL: One or both objects are not Data instances for {resid_label} (Types: {type(g1)}, {type(g2)}).")
         return False
    # Compare node counts first
    if g1.num_nodes != g2.num_nodes:
        print(f"    ❌ FAIL: Node counts differ for {resid_label}: {g1.num_nodes} vs {g2.num_nodes}")
        return False
    # Compare edge counts (important for subsequent checks)
    if g1.num_edges != g2.num_edges:
        print(f"    ❌ FAIL: Edge counts differ for {resid_label}: {g1.num_edges} vs {g2.num_edges}")
        return False

    for attr in tensors_to_check:
        attr_match = True # Track match status for this specific attribute
        has_g1_attr = hasattr(g1, attr)
        has_g2_attr = hasattr(g2, attr)

        if not has_g1_attr and not has_g2_attr:
             # Both missing is okay, especially for edge attrs if no edges
             if attr in ['edge_index', 'edge_s', 'edge_v'] and g1.num_edges == 0:
                 continue
             else: # If non-edge attrs are missing from both, that's odd but technically matching
                 pass
        elif not has_g1_attr or not has_g2_attr:
             print(f"    ❌ FAIL: Attribute '{attr}' missing in one but not both graphs for {resid_label}.")
             match = False
             attr_match = False
             continue # Cannot compare if missing in one

        # If attribute exists in both:
        t1 = getattr(g1, attr)
        t2 = getattr(g2, attr)

        # Handle None case (e.g., edge_index can be None if no edges)
        if t1 is None and t2 is None:
             if attr in ['edge_index', 'edge_s', 'edge_v'] and g1.num_edges == 0:
                 continue # Correctly handled no edges
             else:
                 # If non-edge attrs are None in both, technically matching
                 pass
        elif t1 is None or t2 is None:
            print(f"    ❌ FAIL: Attribute '{attr}' is None in only one graph for {resid_label}.")
            match = False
            attr_match = False
            continue

        # Ensure tensors are compared on CPU
        try:
            t1_cpu = t1.cpu()
            t2_cpu = t2.cpu()
        except Exception as e:
            print(f"    ❌ FAIL: Error moving attribute '{attr}' to CPU for {resid_label}: {e}")
            match = False
            attr_match = False
            continue

        if t1_cpu.shape != t2_cpu.shape:
            print(f"    ❌ FAIL: Shapes differ for '{attr}' for {resid_label}: {t1_cpu.shape} vs {t2_cpu.shape}")
            match = False
            attr_match = False
            continue

        # Perform comparison
        if attr == 'atoms':
            equal = torch.equal(t1_cpu, t2_cpu)
            if not equal: match, attr_match = False, False
        elif attr == 'edge_index':
             # Skip comparison if no edges (shape check already done)
             if t1_cpu.shape[1] == 0: continue
             np_ei1 = np.sort(t1_cpu.numpy(), axis=0)
             np_ei2 = np.sort(t2_cpu.numpy(), axis=0)
             # Ensure sorting is robust for comparison
             if np_ei1.shape[1] > 0:
                  inds1 = np.lexsort((np_ei1[1,:], np_ei1[0,:]))
                  inds2 = np.lexsort((np_ei2[1,:], np_ei2[0,:]))
                  equal = np.array_equal(np_ei1[:, inds1], np_ei2[:, inds2])
             else: # Handle empty edge index case after shape check
                 equal = True

             if equal:
                 # Store sorted indices if they match, needed for feature comparison
                 if np_ei1.shape[1] > 0:
                     g1._sorted_edge_indices = inds1
                     g2._sorted_edge_indices = inds2
                 else: # Handle case of zero edges
                      g1._sorted_edge_indices = np.array([], dtype=int)
                      g2._sorted_edge_indices = np.array([], dtype=int)
             else:
                 match, attr_match = False, False
                 # Avoid adding indices if comparison failed
                 if hasattr(g1, '_sorted_edge_indices'): del g1._sorted_edge_indices
                 if hasattr(g2, '_sorted_edge_indices'): del g2._sorted_edge_indices

        elif attr in ['edge_s', 'edge_v']:
             # Skip comparison if no edges (shape check already done)
             if t1_cpu.shape[0] == 0: continue
             # Check if sorting indices are available and valid
             if hasattr(g1, '_sorted_edge_indices') and hasattr(g2, '_sorted_edge_indices') and \
                g1._sorted_edge_indices is not None and g2._sorted_edge_indices is not None and \
                g1._sorted_edge_indices.size == t1_cpu.shape[0] and g2._sorted_edge_indices.size == t2_cpu.shape[0]:

                 t1_sorted = t1_cpu[g1._sorted_edge_indices]
                 t2_sorted = t2_cpu[g2._sorted_edge_indices]
                 close = torch.allclose(t1_sorted, t2_sorted, rtol=1e-5, atol=1e-8)
                 if not close: match, attr_match = False, False
             else: # edge_index comparison must have failed or indices weren't stored correctly
                 if match: # Only report if edge_index was previously thought to match
                    print(f"    ⚠️ WARN: Cannot compare sorted '{attr}' for {resid_label} as edge_index match failed or indices issue.")
                    match, attr_match = False, False # Treat as a failure if we can't compare features when edges exist

        elif attr == 'x':
             close = torch.allclose(t1_cpu, t2_cpu, rtol=1e-5, atol=1e-8)
             if not close: match, attr_match = False, False
        else: # Should not happen
             try:
                 equal = torch.equal(t1_cpu, t2_cpu)
                 if not equal: match, attr_match = False, False
             except:
                 print(f"    ⚠️ WARN: Could not compare attribute '{attr}' for {resid_label}.")
                 match, attr_match = False, False

        if not attr_match:
             print(f"    ❌ FAIL: Attribute '{attr}' differs for {resid_label}.")

    # Cleanup temporary attributes
    if hasattr(g1, '_sorted_edge_indices'): del g1._sorted_edge_indices
    if hasattr(g2, '_sorted_edge_indices'): del g2._sorted_edge_indices

    return match


# --- Helper: Compare Lists of Graphs ---
# ... (compare_graph_lists function remains the same) ...
def compare_graph_lists(pdb_id, orig_graphs, orig_meta, opt_graphs, opt_meta):
    print(f"\n--- Comparing Generated Graph Lists for PDB: {pdb_id} ---")
    overall_match = True

    if len(orig_graphs) != len(opt_graphs):
        print(f"❌ FAIL: Graph lists have different lengths: {len(orig_graphs)} vs {len(opt_graphs)}")
        return False # Cannot compare further
    print(f" Both lists contain {len(orig_graphs)} graphs.")

    # Create residue maps for metadata alignment
    try:
        # Ensure metadata keys exist before creating map
        if not orig_meta and len(orig_graphs) > 0: # If graphs exist but meta doesn't
            raise ValueError("Original metadata list empty despite graphs present.")
        if not opt_meta and len(opt_graphs) > 0:
            raise ValueError("Optimized metadata list empty despite graphs present.")
        # Allow empty meta if graph lists are also empty
        if orig_meta and not all('chain' in m and 'resid' in m for m in orig_meta):
             raise ValueError("Original metadata list is missing keys ('chain', 'resid').")
        if opt_meta and not all('chain' in m and 'resid' in m for m in opt_meta):
             raise ValueError("Optimized metadata list is missing keys ('chain', 'resid').")

        orig_res_map = {(m['chain'], m['resid']): i for i, m in enumerate(orig_meta)}
        opt_res_map = {(m['chain'], m['resid']): i for i, m in enumerate(opt_meta)}
    except KeyError as e:
         print(f"❌ FAIL: Metadata dictionaries seem to be missing required keys ('chain', 'resid'): {e}")
         return False
    except ValueError as e:
         print(f"❌ FAIL: Problem with metadata structure: {e}")
         return False
    except Exception as e:
         print(f"❌ FAIL: Error creating residue maps from metadata: {e}")
         return False


    common_residues = sorted(list(orig_res_map.keys() & opt_res_map.keys()))

    if len(common_residues) != len(orig_graphs):
        print(f"❌ FAIL: Number of common residues ({len(common_residues)}) doesn't match graph list length ({len(orig_graphs)}). Metadata mismatch?")
        orig_residues_set = set(orig_res_map.keys())
        opt_residues_set = set(opt_res_map.keys())
        print(f"  Original only: {sorted(list(orig_residues_set - opt_residues_set))}")
        print(f"  Optimized only: {sorted(list(opt_residues_set - orig_residues_set))}")
        return False # Metadata itself differs

    if not common_residues and len(orig_graphs) == 0:
         print("✅ OK: Both pathways generated 0 graphs.")
         return True
    elif not common_residues:
         print("⚠️ WARN: No common residues identified between metadata lists (though list lengths matched).")
         return False # Treat as failure if lengths matched but keys didn't

    all_graphs_match = True
    mismatched_residues = []
    print(f" Comparing {len(common_residues)} common residue graphs...")
    for resid_tuple in common_residues:
        orig_idx = orig_res_map[resid_tuple]
        opt_idx = opt_res_map[resid_tuple]
        label = f"{resid_tuple[0]}_{resid_tuple[1]}" # e.g., A_M1

        # Basic check: ensure graphs exist at these indices
        if orig_idx >= len(orig_graphs) or opt_idx >= len(opt_graphs):
             print(f"    ❌ FAIL: Index out of bounds for {label} (Orig: {orig_idx}, Opt: {opt_idx})")
             all_graphs_match = False
             mismatched_residues.append(label + " (Index Error)")
             continue

        graph_match = compare_graph_objects(orig_graphs[orig_idx], opt_graphs[opt_idx], label)
        if not graph_match:
            all_graphs_match = False
            mismatched_residues.append(label)

    if all_graphs_match:
        print(f"✅ SUCCESS: All {len(common_residues)} corresponding graph objects for {pdb_id} appear identical.")
    else:
        print(f"❌ FAIL: Differences found in graph objects for {pdb_id}. Mismatched residues: {mismatched_residues}")

    return all_graphs_match

# --- Helper: Embedding Comparison (Restored) ---
def compare_embeddings(pdb_id, orig_embs_np, orig_meta, opt_embs_np, opt_meta):
    print(f"\n--- Comparing Final Embeddings for PDB: {pdb_id} ---")
    if orig_embs_np is None or opt_embs_np is None:
        print("  One or both embedding arrays are None. Cannot compare.")
        return False
    if orig_meta is None or opt_meta is None:
         print("  One or both metadata lists are None. Cannot compare.")
         return False

    # Basic shape check
    if orig_embs_np.shape != opt_embs_np.shape:
        print(f"❌ FAIL: Embedding shapes differ!")
        print(f" Original shape: {orig_embs_np.shape}")
        print(f" Optimized shape: {opt_embs_np.shape}")
        return False
    else:
        print(f" Embedding shapes match: {orig_embs_np.shape}")

    # Create residue maps
    try:
        orig_res_map = {(m['chain'], m['resid']): i for i, m in enumerate(orig_meta)}
        opt_res_map = {(m['chain'], m['resid']): i for i, m in enumerate(opt_meta)}
    except KeyError:
         print("  ❌ FAIL: Metadata missing 'chain' or 'resid' keys.")
         return False

    common_residues = sorted(list(orig_res_map.keys() & opt_res_map.keys()))

    if not common_residues:
        print("❌ FAIL: No common residues found between the two results based on metadata.")
        return False
    # Verify count matches embedding length (should pass if graph comparison passed)
    if len(common_residues) != orig_embs_np.shape[0]:
         print(f"❌ FAIL: Number of common residues ({len(common_residues)}) doesn't match embedding length ({orig_embs_np.shape[0]}).")
         return False
    else:
        print(f" Found {len(common_residues)} common residues to compare embeddings.")


    # Align embeddings
    aligned_orig = np.array([orig_embs_np[orig_res_map[res]] for res in common_residues])
    aligned_opt = np.array([opt_embs_np[opt_res_map[res]] for res in common_residues])

    # Compare using np.allclose
    are_close = np.allclose(aligned_orig, aligned_opt, rtol=1e-5, atol=1e-8)

    if are_close:
        print("✅ SUCCESS: Aligned embedding arrays are numerically close (np.allclose).")
        return True
    else:
        print("❌ FAIL: Aligned embedding arrays differ significantly.")
        diff = np.abs(aligned_orig - aligned_opt)
        print(f" Max absolute difference: {np.max(diff):.6g}")
        print(f" Mean absolute difference: {np.mean(diff):.6g}")
        # Find first differing element
        indices = np.where(~np.isclose(aligned_orig, aligned_opt, rtol=1e-5, atol=1e-8))
        if len(indices[0]) > 0:
             first_diff_idx = (indices[0][0], indices[1][0])
             first_diff_res_tuple = common_residues[first_diff_idx[0]]
             print(f" First difference at residue index {first_diff_idx[0]} (Chain: {first_diff_res_tuple[0]}, Resid: {first_diff_res_tuple[1]}), element index {first_diff_idx[1]}:")
             print(f"  Original : {aligned_orig[first_diff_idx]:.8f}")
             print(f"  Optimized: {aligned_opt[first_diff_idx]:.8f}")
             print(f"  Difference: {diff[first_diff_idx]:.6g}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Testing PDB directory: {PDB_DIR}")
    print(f"Testing first {NUM_PDBS_TO_TEST} PDB files found.")
    print(f"Include Hets: {INCLUDE_HETS}")
    print(f"Using device: {DEVICE}")

    # Find PDB files
    pdb_files = sorted(glob.glob(os.path.join(PDB_DIR, '*.pdb*')))
    if not pdb_files:
        print(f"Error: No PDB files found in {PDB_DIR}")
        sys.exit(1)

    files_to_process = pdb_files[:NUM_PDBS_TO_TEST]
    print(f"Files to process: {[os.path.basename(f) for f in files_to_process]}")

    # --- Load Model Once ---
    print("\nLoading model...")
    model = initialize_model(CHECKPOINT_PATH, device=DEVICE)
    model.eval() # Ensure model is in evaluation mode
    print("Model loaded.")

    # Instantiate reusable transform components
    original_base_transform = OriginalBaseTransform(edge_cutoff=ORIGINAL_EDGE_CUTOFF, device=DEVICE)
    graph_transform_cpu = GraphPreparationTransformCPU( # This generates graphs on CPU
        include_hets=INCLUDE_HETS,
        env_radius=ENV_RADIUS,
        num_rbf=16
    )
    # Ensure the internal BaseTransform used by the CPU transform for feature calc uses the correct cutoff and DEVICE='cpu'
    graph_transform_cpu.base_transform_cpu = OriginalBaseTransform(edge_cutoff=ORIGINAL_EDGE_CUTOFF, device='cpu')

    # --- Loop through PDB files ---
    overall_graph_match = True
    overall_embedding_match = True
    for pdb_file_path in files_to_process:
        pdb_id = os.path.basename(pdb_file_path).split('.')[0]
        print(f"\n{'='*20} Processing PDB: {pdb_id} {'='*20}")

        # --- Initialize results for this PDB ---
        orig_graphs_list = None
        orig_metadata_list = None
        opt_graphs_list = None
        opt_metadata_list = None
        orig_embs_np = None
        opt_embs_np = None
        pdb_graph_match = False
        pdb_embedding_match = False

        # --- 1. Original Path: Graph Gen + Inference ---
        print("\n--- Running Original Path ---")
        start_time_orig = time.time()
        try:
            atom_df_orig = process_pdb(pdb_file_path, include_hets=INCLUDE_HETS)
            if atom_df_orig is None or atom_df_orig.empty:
                 print("  ⚠️ process_pdb failed. Skipping PDB.")
                 continue

            # Generate Graphs
            current_orig_graphs = []
            current_orig_meta = []
            for (c, i, r), res_df in atom_df_orig.groupby(['chain', 'residue', 'resname']):
                if r not in atom_info.aa[:20]: continue
                resid_letter = atom_info.aa_to_letter(r)
                resid_str = resid_letter + str(i)
                out_tuple = extract_env_from_resid_original(
                                        atom_df_orig, (c, resid_str),
                                        env_radius=ENV_RADIUS, res_df=res_df.copy(), train_mode=False)
                if out_tuple and isinstance(out_tuple[0], Data):
                    graph_obj = out_tuple[0]
                    graph_obj.protein_id = pdb_id
                    graph_obj.resid = resid_str
                    graph_obj.chain = c
                    current_orig_graphs.append(graph_obj)
                    confidence = res_df['bfactor'].iloc[0] if 'bfactor' in res_df.columns else 0.0
                    current_orig_meta.append({
                        'protein_id': pdb_id, 'chain': c,
                        'resid': resid_str, 'confidence': confidence
                    })
            orig_graphs_list = current_orig_graphs
            orig_metadata_list = current_orig_meta
            print(f"  Original path generated {len(orig_graphs_list)} graphs.")

            # Inference
            if orig_graphs_list:
                 graph_batch_orig = Batch.from_data_list(orig_graphs_list).to(DEVICE)
                 with torch.no_grad():
                     embs_tensor_orig, _ = model.online_encoder(graph_batch_orig, return_projection=False)
                     orig_embs_np = embs_tensor_orig.float().cpu().numpy()
                 if len(orig_metadata_list) != orig_embs_np.shape[0]:
                      print(f"  ⚠️ Original inference meta/emb count mismatch: {len(orig_metadata_list)} vs {orig_embs_np.shape[0]}")
                      orig_embs_np = None # Invalidate embeddings
            else:
                 print("  No original graphs generated, skipping inference.")

        except Exception as e:
            print(f"  ❌ Error during original path for {pdb_id}: {e}")
            import traceback; traceback.print_exc()
        end_time_orig = time.time()
        print(f"  Original path took {end_time_orig - start_time_orig:.2f}s.")

        # --- 2. Optimized Path: Graph Gen + Inference ---
        print("\n--- Running Optimized Path ---")
        start_time_opt = time.time()
        try:
            # Preprocess
            raw_atoms = process_pdb(pdb_file_path)
            if raw_atoms is None or raw_atoms.empty:
                 print("  ⚠️ process_pdb failed. Skipping PDB.")
                 continue
            atom_df_opt = first_model_filter(raw_atoms)
            atom_df_opt = atom_df_opt[~atom_df_opt.hetero.str.contains('W', na=False)]
            atom_df_opt = atom_df_opt[atom_df_opt['element'] != 'H']
            if not INCLUDE_HETS:
                atom_df_opt = atom_df_opt[atom_df_opt.resname.isin(atom_info.aa)]
            atom_df_opt = atom_df_opt.reset_index(drop=True)
            atom_df_opt['id'] = pdb_id
            if atom_df_opt.empty:
                print("  ⚠️ DataFrame empty after filtering. Skipping PDB.")
                continue

            # Generate Graphs
            transformed_item = graph_transform_cpu({'atoms': atom_df_opt, 'id': pdb_id})
            if transformed_item is None:
                 print(f"  ⚠️ GraphPreparationTransformCPU returned None. Skipping PDB.")
                 continue
            if 'graphs' not in transformed_item or 'metadata' not in transformed_item:
                 print(f"  ⚠️ Optimized transform output missing keys. Skipping PDB.")
                 continue
            opt_graphs_list = transformed_item['graphs']
            opt_metadata_list = transformed_item['metadata']
            print(f"  Optimized path generated {len(opt_graphs_list)} graphs.")

            # Inference
            if opt_graphs_list:
                graph_batch_opt = Batch.from_data_list(opt_graphs_list).to(DEVICE)
                with torch.no_grad():
                    embs_tensor_opt, _ = model.online_encoder(graph_batch_opt, return_projection=False)
                    opt_embs_np = embs_tensor_opt.float().cpu().numpy()
                if len(opt_metadata_list) != opt_embs_np.shape[0]:
                     print(f"  ⚠️ Optimized inference meta/emb count mismatch: {len(opt_metadata_list)} vs {opt_embs_np.shape[0]}")
                     opt_embs_np = None # Invalidate embeddings
            else:
                 print("  No optimized graphs generated, skipping inference.")

        except Exception as e:
            print(f"  ❌ Error during optimized path for {pdb_id}: {e}")
            import traceback; traceback.print_exc()
        end_time_opt = time.time()
        print(f"  Optimized path took {end_time_opt - start_time_opt:.2f}s.")

        # --- 3. Compare Graphs for this PDB ---
        if orig_graphs_list is not None and opt_graphs_list is not None:
            pdb_graph_match = compare_graph_lists(pdb_id, orig_graphs_list, orig_metadata_list, opt_graphs_list, opt_metadata_list)
            if not pdb_graph_match:
                overall_graph_match = False
        else:
             print(f" Skipping graph comparison for {pdb_id} due to generation errors.")
             overall_graph_match = False

        # --- 4. Compare Embeddings for this PDB ---
        if pdb_graph_match: # Only compare embeddings if graphs matched
             if orig_embs_np is not None and opt_embs_np is not None:
                 pdb_embedding_match = compare_embeddings(pdb_id, orig_embs_np, orig_metadata_list, opt_embs_np, opt_metadata_list)
                 if not pdb_embedding_match:
                      overall_embedding_match = False
             else:
                  print(f" Skipping embedding comparison for {pdb_id} due to missing embeddings from one or both paths (check errors above).")
                  overall_embedding_match = False # Mark as mismatch if embeddings couldn't be generated/compared
        else:
            print(f" Skipping embedding comparison for {pdb_id} because graphs did not match.")
            overall_embedding_match = False


    # --- Final Summary ---
    print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    if overall_graph_match:
        print(f"✅ Graph Generation: Consistent for all {len(files_to_process)} tested PDBs.")
    else:
        print(f"❌ Graph Generation: Differences detected for one or more of the {len(files_to_process)} tested PDBs.")

    if overall_embedding_match:
        print(f"✅ Embedding Results: Consistent (numerically close) for all tested PDBs where graphs matched.")
    else:
        print(f"❌ Embedding Results: Differences detected (or comparisons skipped) for one or more tested PDBs.")

    if overall_graph_match and overall_embedding_match:
         print("\nCONCLUSION: Both graph generation and inference appear consistent within GPU numerical precision.")
    elif overall_graph_match and not overall_embedding_match:
         print("\nCONCLUSION: Graph generation is consistent, but embedding results differ beyond expected numerical noise. Check inference steps.")
    else:
         print("\nCONCLUSION: Graph generation showed inconsistencies. Embedding differences are expected.")
