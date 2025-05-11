import torch
import numpy as np
import pandas as pd
import scipy.spatial
from torch_geometric.data import Data
import torch_cluster
import traceback  # Add for better error handling
from multiprocessing import shared_memory # ADDED for shared memory
import time # ADDED for timing in workers

# Import necessary functions from embedding_utils
try:
    from collapse.embedding_utils import (
        _element_mapping, _normalize, _rbf, _edge_features,
        sample_functional_center
    )
except ImportError as e:
    print(f"Error importing from embedding_utils: {e}")
    raise

def create_pyg_graph(df_env, edge_cutoff=4.5, num_rbf=16, device=torch.device('cpu')):
    """
    Creates a PyG Data object from a DataFrame representing a residue's environment on CPU.
    """
    graph_device = torch.device('cpu')

    try:
        with torch.no_grad():
            # --- Node Features ---
            if 'element' not in df_env.columns: 
                print("Warning: 'element' column not found in DataFrame for create_pyg_graph")
                return None
                
            atoms = torch.as_tensor([_element_mapping(e) for e in df_env['element'].values],
                                    dtype=torch.long, device=graph_device)

            # Coordinates
            if not {'x', 'y', 'z'}.issubset(df_env.columns): 
                print("Warning: Coordinate columns not found in DataFrame for create_pyg_graph")
                return None
                
            coords = torch.as_tensor(df_env[['x', 'y', 'z']].to_numpy(dtype=np.float32),
                                     dtype=torch.float32, device=graph_device)

            # Basic validation
            if coords.dim() != 2 or coords.shape[1] != 3: 
                print(f"Warning: Invalid coordinates shape: {coords.shape} for create_pyg_graph")
                return None
            if coords.shape[0] == 0: 
                print("Warning: Empty coordinates for create_pyg_graph")
                return None

            # --- Edge Index ---
            edge_index = torch_cluster.radius_graph(
                coords,
                r=edge_cutoff,
                max_num_neighbors=coords.shape[0]
            )

            # --- Edge Features ---
            if edge_index.shape[1] > 0:
                edge_s, edge_v = _edge_features(coords, edge_index, D_max=edge_cutoff,
                                                num_rbf=num_rbf, device=graph_device)
            else:
                edge_s = torch.empty((0, num_rbf), dtype=torch.float32, device=graph_device)
                edge_v = torch.empty((0, 1, 3), dtype=torch.float32, device=graph_device)

            # --- Create Data Object ---
            data = Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
            return data

    except Exception as e:
        print(f"Error during PyG graph creation (CPU): {e}")
        print(traceback.format_exc())
        return None

def _process_residue_env_worker(residue_key, 
                                protein_all_coords_np, # NumPy array of all atom coordinates
                                protein_all_elements_np, # NumPy array of all atom elements
                                # chain_atom_dfs_global is still needed for sample_functional_center if it expects a DataFrame
                                chain_atom_dfs_global, \
                                kdtree_global, \
                                env_radius):
    """
    Worker function to identify a single residue's environment atoms and return their data as NumPy arrays.
    """
    worker_process_start_time = time.perf_counter()
    print(f"WORKER {residue_key[0]}-{residue_key[2] if len(residue_key) > 2 else 'short_key'}: ALIVE AND STARTING _process_residue_env_worker") # MODIFIED: Uncommented and made safer
    try:
        protein_id, chain_id, resnum, resname_letter, bfactor = residue_key
        resid_tuple = (resname_letter, resnum)

        s_time = time.perf_counter()
        chain_atoms_df = chain_atom_dfs_global.get(chain_id)
        if chain_atoms_df is not None: print(f"WORKER {resid_tuple}: Got chain_df. Size: {len(chain_atoms_df) if chain_atoms_df is not None else 'N/A'}. Time: {time.perf_counter() - s_time:.6f}s") # MODIFIED
        else: print(f"WORKER {resid_tuple}: Chain DF not found for chain {chain_id}") # MODIFIED

        if chain_atoms_df is None:
            # print(f"Warning: No data found for chain {chain_id} in worker for {resid_tuple}")
            return None, None

        s_time = time.perf_counter()
        # Filter the chain-specific DataFrame to get atoms of the target residue
        # Ensure resnum is the correct type for comparison with 'residue' column
        # If 'residue' column in chain_atoms_df is string, ensure resnum is string, or vice-versa for numeric.
        # Assuming chain_atoms_df['residue'] is numeric, as is typical.
        res_df = chain_atoms_df[chain_atoms_df['residue'] == int(resnum)] 
        if not res_df.empty: print(f"WORKER {resid_tuple}: Got res_df. Size: {len(res_df)}. Time: {time.perf_counter() - s_time:.6f}s") # MODIFIED
        else: print(f"WORKER {resid_tuple}: Res_df empty for resnum {resnum} (type: {type(resnum)}) in chain {chain_id}") # MODIFIED

        if res_df.empty:
            # print(f"Warning: No atoms found for residue {resnum} in chain {chain_id} in worker for {resid_tuple}")
            return None, None

        s_time = time.perf_counter()
        center = sample_functional_center(res_df, resid_tuple, train_mode=False)
        if center is not None: print(f"WORKER {resid_tuple}: Center calculated {center}. Time: {time.perf_counter() - s_time:.6f}s") # MODIFIED
        else: print(f"WORKER {resid_tuple}: Center calculation failed") # MODIFIED

        if center is None:
            # print(f"Warning: Could not calculate center for residue {resid_tuple} in worker")
            return None, None

        s_time = time.perf_counter()
        pt_indices = kdtree_global.query_ball_point(center, r=env_radius)
        print(f"WORKER {resid_tuple}: KDTree query_ball_point. Num_indices: {len(pt_indices) if pt_indices is not None else 'N/A'}. Time: {time.perf_counter() - s_time:.6f}s") # MODIFIED
        
        if not pt_indices or len(pt_indices) == 0:
            print(f"WORKER {resid_tuple}: No KDTree neighbors found.") # MODIFIED
            return None, None # No environment found

        # ---- ADDED: Rigorous check of pt_indices ----
        print(f"WORKER {resid_tuple}: type(pt_indices): {type(pt_indices)}")
        if isinstance(pt_indices, list) and len(pt_indices) > 0:
            print(f"WORKER {resid_tuple}: type(pt_indices[0]): {type(pt_indices[0])}")
            # Convert to numpy array for min/max if it's a list of numbers, handle potential non-numeric gracefully
            try:
                pt_indices_np = np.array(pt_indices)
                print(f"WORKER {resid_tuple}: pt_indices min: {np.min(pt_indices_np)}, max: {np.max(pt_indices_np)}")
            except TypeError as e:
                print(f"WORKER {resid_tuple}: pt_indices contains non-numeric data, cannot get min/max easily. Error: {e}")
                print(f"WORKER {resid_tuple}: First 10 pt_indices: {pt_indices[:10]}")
        
        print(f"WORKER {resid_tuple}: protein_all_coords_np.shape: {protein_all_coords_np.shape}")
        # ---- END ADDED ----

        s_time = time.perf_counter()
        # Extract environment data as NumPy arrays
        # Test basic slicing first
        try:
            test_slice = protein_all_coords_np[0:1]
            print(f"WORKER {resid_tuple}: Test slice successful. Shape: {test_slice.shape}")
        except Exception as e:
            print(f"WORKER {resid_tuple}: Test slice FAILED: {e}")
            print(traceback.format_exc())
            return None, None # Cannot proceed if basic slicing fails

        # ---- MODIFIED: Test variants of pt_indices slicing ----
        print(f"WORKER {resid_tuple}: Attempting advanced indexing with original pt_indices (length {len(pt_indices)})...")
        try:
            env_coords_np_original = protein_all_coords_np[pt_indices] # This is the line that seems to hang/crash
            print(f"WORKER {resid_tuple}: Slicing with original pt_indices SUCCESSFUL. Shape: {env_coords_np_original.shape}")
            
            # If original works, proceed with it and also slice elements
            env_coords_np = env_coords_np_original
            env_elements_np = protein_all_elements_np[pt_indices]
            print(f"WORKER {resid_tuple}: Also sliced elements with original pt_indices.")

        except Exception as e:
            print(f"WORKER {resid_tuple}: Slicing with original pt_indices FAILED: {e}")
            print(traceback.format_exc())

            print(f"WORKER {resid_tuple}: Attempting advanced indexing with pt_indices.copy()...")
            try:
                pt_indices_copy = pt_indices.copy() # Try with a copy of the list
                env_coords_np_copied_list = protein_all_coords_np[pt_indices_copy]
                print(f"WORKER {resid_tuple}: Slicing with pt_indices.copy() SUCCESSFUL. Shape: {env_coords_np_copied_list.shape}")
                env_coords_np = env_coords_np_copied_list
                env_elements_np = protein_all_elements_np[pt_indices_copy]
                print(f"WORKER {resid_tuple}: Also sliced elements with pt_indices.copy().")
            except Exception as e_copy:
                print(f"WORKER {resid_tuple}: Slicing with pt_indices.copy() FAILED: {e_copy}")
                print(traceback.format_exc())

                print(f"WORKER {resid_tuple}: Attempting advanced indexing with np.array(pt_indices)...")
                try:
                    pt_indices_np_array = np.array(pt_indices)
                    env_coords_np_np_array_indices = protein_all_coords_np[pt_indices_np_array]
                    print(f"WORKER {resid_tuple}: Slicing with np.array(pt_indices) SUCCESSFUL. Shape: {env_coords_np_np_array_indices.shape}")
                    env_coords_np = env_coords_np_np_array_indices
                    env_elements_np = protein_all_elements_np[pt_indices_np_array]
                    print(f"WORKER {resid_tuple}: Also sliced elements with np.array(pt_indices).")
                except Exception as e_np_array:
                    print(f"WORKER {resid_tuple}: Slicing with np.array(pt_indices) FAILED: {e_np_array}")
                    print(traceback.format_exc())
                    print(f"WORKER {resid_tuple}: All slicing attempts failed. Returning None.")
                    return None, None # All attempts failed
        # ---- END MODIFIED ----

        # Ensure env_coords_np and env_elements_np are defined one way or another if an attempt succeeded
        if 'env_coords_np' not in locals() or 'env_elements_np' not in locals():
            print(f"WORKER {resid_tuple}: env_coords_np or env_elements_np not defined after slicing attempts. This should not happen if an attempt succeeded.")
            return None, None

        print(f"WORKER {resid_tuple}: Extracted env numpy arrays. Coords shape: {env_coords_np.shape}. Time: {time.perf_counter() - s_time:.6f}s") # MODIFIED
        
        env_data_for_gpu = (env_coords_np, env_elements_np)

        metadata = {
            'protein_id': protein_id,
            'chain': chain_id,
            'resid': f"{resname_letter}{resnum}",
            'resname_letter': resname_letter,
            'resnum': resnum,
            'confidence': bfactor
        }
        print(f"WORKER {resid_tuple}: Processed successfully. Attempting to return. Total time before return: {time.perf_counter() - worker_process_start_time:.6f}s")
        # MODIFIED: Temporarily return only metadata to test pickling of NumPy arrays
        # return metadata, env_data_for_gpu 
        # return metadata, None # TRY THIS FIRST -- This seemed to hang.
        
        # TRY THIS SECOND: Return a very simple, fresh object
        simple_return_value = ("worker_ok", residue_key[2]) # e.g., ("worker_ok", resnum)
        # print(f"WORKER {resid_tuple}: Attempting to return simple value: {simple_return_value}")
        return simple_return_value

        # If the above works, then try returning simple arrays not from shared memory:
        # fake_coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        # fake_elements = np.array(['C'], dtype=object)
        # return metadata, (fake_coords, fake_elements) # TRY THIS THIRD

    except Exception as e:
        print(f"Error in _process_residue_env_worker for {residue_key}: {e}")
        print(traceback.format_exc())
        return None, None

def global_worker_task_executor(args_tuple):
    """
    Top-level wrapper function for multiprocessing.
    """
    executor_start_time = time.perf_counter()
    print(f"EXECUTOR for {args_tuple[0][0]}-{args_tuple[0][2] if len(args_tuple[0]) > 2 else 'short_key_args'}: ALIVE AND STARTING global_worker_task_executor") # MODIFIED: Uncommented and made safer
    try:
        # Unpack all arguments including shared memory identifiers
        residue_key, \
            shm_coords_name, coords_shape, coords_dtype, \
            shm_elements_name, elements_shape, elements_dtype, \
            chain_atom_dfs_global, kdtree_global, env_radius = args_tuple
        
        # --- ADDED: Reconstruct NumPy arrays from shared memory ---
        shm_attach_start_time = time.perf_counter()
        existing_shm_coords = shared_memory.SharedMemory(name=shm_coords_name)
        protein_all_coords_np = np.ndarray(coords_shape, dtype=coords_dtype, buffer=existing_shm_coords.buf)

        existing_shm_elements = shared_memory.SharedMemory(name=shm_elements_name)
        protein_all_elements_np = np.ndarray(elements_shape, dtype=elements_dtype, buffer=existing_shm_elements.buf)
        # print(f"EXECUTOR for {residue_key[0]}-{residue_key[2]}: Attached to SHM. Time: {time.perf_counter() - shm_attach_start_time:.6f}s")
        # --- END ADDED ---
        
        result = _process_residue_env_worker(\
            residue_key,\
            protein_all_coords_np,\
            protein_all_elements_np,\
            chain_atom_dfs_global,\
            kdtree_global,\
            env_radius\
        )
        # print(f"EXECUTOR for {residue_key[0]}-{residue_key[2]}: Worker call done. Total time: {time.perf_counter() - executor_start_time:.6f}s")
        return result
    except Exception as e:
        print(f"Error in global_worker_task_executor for {args_tuple[0] if args_tuple and len(args_tuple) > 0 else 'unknown_task'}: {e}") # Improved error logging
        print(traceback.format_exc())
        return None, None 
    finally:
        # --- ADDED: Ensure worker closes its connection to shared memory ---
        # final_close_time = time.perf_counter()
        if existing_shm_coords:
            existing_shm_coords.close()
        if existing_shm_elements:
            existing_shm_elements.close()
        # print(f"EXECUTOR for {args_tuple[0][0]}-{args_tuple[0][2] if args_tuple and len(args_tuple) > 0 and len(args_tuple[0]) > 2 else 'unknown_task_finally'}: Closed SHM. Close time: {time.perf_counter() - final_close_time:.6f}s") # MODIFIED: Made safer
        # --- END ADDED --- 