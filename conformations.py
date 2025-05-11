import os
import time
import argparse # Keep for potential future CLI use, though not primary in notebook
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # Suppress SettingWithCopyWarning

import torch
import torch.nn as nn # Needed if we re-wrap model? No, initialize_model does it.
import torch_geometric
from torch_geometric.data import Data, Batch
import torch_cluster # Needed for radius_graph
import scipy.spatial # For cKDTree

# For parallel processing
import multiprocessing  # Keep the original import
from multiprocessing import shared_memory # ADDED for shared memory
from functools import partial
import traceback # For printing detailed errors
import cProfile # ADDED: For programmatic profiling
import pstats   # ADDED: For programmatic profiling

import ray # ADDED
import psutil # ADDED for system memory information
import shutil # ADDED for directory operations

import atexit # ADDED for memmap cleanup

# --- Imports from project ---
# Assuming collapse and embedding_utils are importable from the notebook's environment
# Adjust paths if necessary
try:
    from collapse import atom_info, initialize_model # initialize_model loads model + BYOL wrapper
    from atom3d.filters.filters import first_model_filter # Preprocessing
    import atom3d.util.formats as fo # For reading PDB

    # Import necessary functions directly from embedding_utils
    from collapse.embedding_utils import (
        _element_mapping, _normalize, _rbf, _edge_features,
        sample_functional_center, # Uses atom_info
        ELEMENT_MAPPING, DEFAULT_ELEMENT # Constants
    )
    # Add atom_info.aa_to_letter_dict if it exists and is useful
    if hasattr(atom_info, 'aa_to_letter'):
        # Precompute dict for faster mapping if not already done in atom_info
        if not hasattr(atom_info, 'aa_to_letter_dict'):
             # Assuming atom_info.aa contains 3-letter codes and atom_info.aa_abbr contains 1-letter
             if hasattr(atom_info, 'aa') and hasattr(atom_info, 'aa_abbr') and len(atom_info.aa) == len(atom_info.aa_abbr):
                  atom_info.aa_to_letter_dict = dict(zip(atom_info.aa, atom_info.aa_abbr))
             else:
                  # Fallback or raise error if mapping cannot be constructed
                  print("Warning: Could not reliably construct atom_info.aa_to_letter_dict.")
                  # Use the function call as fallback in _process_residue_env_worker
                  atom_info.aa_to_letter_dict = None # Signal to use the function
    else:
        raise AttributeError("atom_info.aa_to_letter function/mapping missing.")

except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'collapse', 'atom3d', and 'embedding_utils.py' are accessible.")
    # Add fallback dummy functions or raise error if crucial parts are missing
    raise(e) # Re-raise error to stop execution if imports fail

from collapse.worker_functions import global_worker_task_executor, _process_residue_env_worker # We will adapt its usage

# --- Try to set start method as early as possible ---
# Ensure this is called before any torch imports if they are not at the top, 
# and definitely before any CUDA-related operations in the main process that might influence spawned children.
if __name__ == '__main__': # Guarding this, though it's often at top level
    try:
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            print(f"Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        # Might fail if already set and force=False, or if in an environment where it cannot be changed.
        # Or if called too late after context has been used.
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            print(f"Warning: Could not force multiprocessing start method to 'spawn' ('{e}'). Using default: '{current_method}'.")
        # else: print(f"Multiprocessing start method already '{current_method}'.") # Less verbose now
    except Exception as e: # Catch other potential exceptions like AttributeError in restricted envs
        print(f"Warning: Exception while trying to set multiprocessing start method: {e}")
# --- End set start method ---

# --- ADDED: Top-level worker_init_fn for pickling --- 
def worker_init_fn():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # print(f"Worker {os.getpid()} initialized, CUDA_VISIBLE_DEVICES set to ''") # Optional: for debugging init
# --- END ADDED ---

# Temporary file paths for memmap
MEMMAP_DIR = os.path.join(os.environ.get('HOME', '/tmp'), 'collapse_ray_memmap')
os.makedirs(MEMMAP_DIR, exist_ok=True)
MEMMAP_COORDS_PATH = os.path.join(MEMMAP_DIR, "collapse_coords.dat")
MEMMAP_ELEMENTS_PATH = os.path.join(MEMMAP_DIR, "collapse_elements.dat")
MEMMAP_RESIDUES_PATH = os.path.join(MEMMAP_DIR, "collapse_residues.dat") # For residue numbers
MEMMAP_CHAINS_PATH = os.path.join(MEMMAP_DIR, "collapse_chains.dat")   # For chain IDs
MEMMAP_ATOM_NAMES_PATH = os.path.join(MEMMAP_DIR, "collapse_atom_names.dat") # For atom names (needed by sample_functional_center)
MEMMAP_ELEMENT_DICT_PATH = os.path.join(MEMMAP_DIR, "collapse_element_dict.dat")
MEMMAP_CHAIN_DICT_PATH = os.path.join(MEMMAP_DIR, "collapse_chain_dict.dat")
MEMMAP_ATOM_NAME_DICT_PATH = os.path.join(MEMMAP_DIR, "collapse_atom_name_dict.dat")

# --- ADDED: atexit cleanup for memmap files ---
MEMMAP_FILES_TO_CLEAN = [MEMMAP_COORDS_PATH, MEMMAP_ELEMENTS_PATH, MEMMAP_RESIDUES_PATH, MEMMAP_CHAINS_PATH, MEMMAP_ATOM_NAMES_PATH, MEMMAP_ELEMENT_DICT_PATH, MEMMAP_CHAIN_DICT_PATH, MEMMAP_ATOM_NAME_DICT_PATH]

def cleanup_memmap_files_on_exit():
    print(f"ATEXIT: Cleaning up {len(MEMMAP_FILES_TO_CLEAN)} memmap files...")
    for p in MEMMAP_FILES_TO_CLEAN:
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"  ATEXIT: Removed {p}")
            except Exception as e_remove:
                print(f"  ATEXIT: Error removing {p}: {e_remove}")
    
    # Try to remove the directory - use shutil to handle non-empty directory
    try:
        if os.path.exists(MEMMAP_DIR):
            shutil.rmtree(MEMMAP_DIR)
            print(f"  ATEXIT: Removed directory and contents {MEMMAP_DIR}")
    except Exception as e_dir:
        print(f"  ATEXIT: Error removing directory {MEMMAP_DIR}: {e_dir}")
    
    # If Ray is initialized, shut it down properly
    if ray.is_initialized():
        try:
            ray.shutdown()
            print("  ATEXIT: Ray shutdown completed")
        except Exception as e_ray:
            print(f"  ATEXIT: Error during Ray shutdown: {e_ray}")

atexit.register(cleanup_memmap_files_on_exit)
# --- END ADDED ---

# %% [markdown]
# ## Configuration

# %%
# --- Parameters ---
# <<< USER: SET YOUR PDB FILE HERE >>>
PDB_FILE_PATH = "/scratch/groups/rbaltman/ziyiw23/exp_conform/MGYP000002588102.pdb"
# Checkpoint for the COLLAPSE model
CHECKPOINT_PATH = 'data/checkpoints/collapse_base.pt'
# Radius to define the local environment around each residue's center
ENV_RADIUS = 10.0
# Distance cutoff for creating edges within the residue environment graph
EDGE_CUTOFF = 4.5
# Number of radial basis functions used for edge features
NUM_RBF = 16
# Whether to include non-standard residues/heteroatoms in the graph generation
INCLUDE_HETS = False # Typically False for canonical embeddings
# Enable torch.compile for potential model acceleration (requires PyTorch 2.0+)
COMPILE_MODEL = True
# Number of CPU workers for parallel graph generation. Adjust based on your system.
NUM_WORKERS = max(1, os.cpu_count() // 2)

# %% [markdown]
# ## Device Setup

# %%
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Moved to main
# print(f"Using device: {device}") # Moved to main
# if torch.cuda.is_available(): # Moved to main
#     print(f"Found {torch.cuda.device_count()} CUDA devices.") # Moved to main

# Set multiprocessing start method (important for CUDA compatibility with multiprocessing)
# 'spawn' is generally recommended when using CUDA.
# try:
#     if multiprocessing.get_start_method(allow_none=True) != 'spawn':
#         multiprocessing.set_start_method('spawn', force=True)
#         print(f"Set multiprocessing start method to 'spawn'.")
# except Exception as e:
#     # Ignore if it's already 'spawn' or cannot be changed (e.g., in certain envs)
#     current_method = multiprocessing.get_start_method(allow_none=True)
#     if current_method != 'spawn':
#         print(f"Warning: Could not force multiprocessing start method to 'spawn' ('{e}'). Using default: '{current_method}'.")


# %% [markdown]
# ## Model Loading and Compilation

# %%
print("Loading model...")
# Use initialize_model from gen_embed (implicitly imported via collapse) which correctly sets up BYOL wrapper
# --- MODIFICATION: Load the model initially onto CPU ---
cpu_device = torch.device('cpu')
model = initialize_model(CHECKPOINT_PATH, device=cpu_device)
# --- END MODIFICATION ---
model.eval() # Set the model to evaluation mode (disables dropout, etc.)
print("Model loaded onto CPU.")

# DataParallel is generally not needed or beneficial for single PDB inference
# if torch.cuda.device_count() > 1:
#     print(f"Wrapping model with nn.DataParallel for {torch.cuda.device_count()} GPUs.")
#     model = nn.DataParallel(model)

# --- MODIFICATION: Compile step moved after model is moved to GPU ---
# if COMPILE_MODEL and hasattr(torch, 'compile'):
#     print("Compiling model (this may take a moment)...")
#     try:
#         # 'reduce-overhead' is often good for inference speed with relatively static input shapes
#         model = torch.compile(model, mode="reduce-overhead")
#         print("Model compiled successfully.")
#     except Exception as e:
#         print(f"Warning: Model compilation failed: {e}")
# else:
#     if not hasattr(torch, 'compile'):
#         print("torch.compile not available (requires PyTorch 2.0+).")
#     print("Model compilation skipped.")
# --- END MODIFICATION ---


# %% [markdown]
# ## Helper Functions (Adapted from `embedding_utils.py`)

# %%
# --- Graph Construction Helpers ---
# Essential functions like _element_mapping, _normalize, _rbf, _edge_features
# are assumed to be imported correctly from embedding_utils.py

def create_pyg_graph(df_env, edge_cutoff=EDGE_CUTOFF, num_rbf=NUM_RBF, device=torch.device('cpu')):
    """
    Creates a PyG Data object from a DataFrame representing a residue's environment.
    Simplified and adapted from embedding_utils.BaseTransform.
    Operates on CPU. Graphs will be moved to GPU in batch later.
    """
    graph_device = torch.device('cpu')

    try:
        with torch.no_grad():
            # --- Node Features ---
            # Element mapping
            if 'element' not in df_env.columns: return None
            # Use list comprehension for potentially faster mapping than pd.Series.map
            atoms = torch.as_tensor([_element_mapping(e) for e in df_env['element'].values],
                                    dtype=torch.long, device=graph_device)

            # Coordinates
            if not {'x', 'y', 'z'}.issubset(df_env.columns): return None
            coords = torch.as_tensor(df_env[['x', 'y', 'z']].to_numpy(dtype=np.float32),
                                     dtype=torch.float32, device=graph_device)

            # Basic validation
            if coords.dim() != 2 or coords.shape[1] != 3: return None # Expect N x 3
            if coords.shape[0] == 0: return None # Empty environment

            # --- Edge Index ---
            # Use torch_cluster.radius_graph to find edges within the cutoff distance
            edge_index = torch_cluster.radius_graph(
                coords,
                r=edge_cutoff,
                max_num_neighbors=coords.shape[0] # Ensure all potential neighbors considered
            )

            # --- Edge Features ---
            if edge_index.shape[1] > 0:
                # Calculate scalar (rbf) and vector features for edges
                edge_s, edge_v = _edge_features(coords, edge_index, D_max=edge_cutoff,
                                                num_rbf=num_rbf, device=graph_device)
            else:
                # Handle cases with no edges (e.g., isolated atoms in environment)
                # Create empty tensors with correct dimensions
                edge_s = torch.empty((0, num_rbf), dtype=torch.float32, device=graph_device)
                edge_v = torch.empty((0, 1, 3), dtype=torch.float32, device=graph_device)

            # --- Create Data Object ---
            data = Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
            return data

    except Exception as e:
        # Catch potential errors during graph conversion (e.g., tensor issues)
        print(f"Error during PyG graph creation: {e}\n{traceback.format_exc()}")
        return None

# --- Residue Processing Helper ---
# sample_functional_center is assumed imported from embedding_utils

def create_pyg_graph_on_gpu(env_coords_np, env_elements_np, 
                            target_device, 
                            edge_cutoff=EDGE_CUTOFF, # Use global config
                            num_rbf=NUM_RBF):      # Use global config
    """
    Creates a PyG Data object directly on the target_device (GPU)
    from NumPy arrays representing a single residue's environment.
    """
    try:
        with torch.no_grad():
            # Convert NumPy arrays to PyTorch tensors on the target_device
            coords = torch.from_numpy(env_coords_np).float().to(target_device)
            # _element_mapping is CPU-bound, so map first, then convert to tensor
            mapped_elements = [_element_mapping(e) for e in env_elements_np]
            atoms = torch.tensor(mapped_elements, dtype=torch.long, device=target_device)

            if coords.dim() != 2 or coords.shape[1] != 3: 
                # print(f"Debug: Invalid env_coords_np shape: {env_coords_np.shape} for GPU graph") # Less verbose
                return None
            if coords.shape[0] == 0: 
                # print("Debug: Empty env_coords_np for GPU graph") # Less verbose
                return None

            edge_index = torch_cluster.radius_graph(
                coords, r=edge_cutoff, max_num_neighbors=coords.shape[0]
            ) # This runs on target_device as coords is on target_device

            if edge_index.shape[1] > 0:
                edge_s, edge_v = _edge_features(coords, edge_index, D_max=edge_cutoff,
                                                num_rbf=num_rbf, device=target_device)
            else:
                edge_s = torch.empty((0, num_rbf), dtype=torch.float32, device=target_device)
                edge_v = torch.empty((0, 1, 3), dtype=torch.float32, device=target_device)

            data = Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
            return data
    except Exception as e:
        print(f"Error during PyG graph creation on GPU: {e}")
        print(traceback.format_exc())
        return None

# %% [markdown]
# ## Ray Actor Definition and Worker Wrapper
# %%

# This wrapper will be called by the Ray actor.
# It adapts the call to the existing _process_residue_env_worker.
def process_residue_for_ray(residue_key_as_task, # (protein_id, chain_id, resnum, resname_letter, bfactor)
                             coords_memmap_path,
                             elements_memmap_path,
                             residues_memmap_path, # for res_df reconstruction
                             chains_memmap_path,   # for res_df reconstruction
                             atom_names_memmap_path, # for res_df reconstruction
                             full_coords_shape,    # for memmap
                             full_elements_shape,  # for memmap
                             full_residues_shape,
                             full_chains_shape,
                             full_atom_names_shape,
                             element_lookup, chain_lookup, atom_name_lookup, # Dictionaries for code->string conversion
                             env_radius_val,
                             build_kdtree_fn): # Function to build KDTree
    """
    Wrapper to be called by Ray.
    Adapts arguments for _process_residue_env_worker.
    Constructs a minimal res_df for the target residue for sample_functional_center.
    """
    try:
        # Check if all memmap files exist before opening
        for path in [coords_memmap_path, elements_memmap_path, residues_memmap_path, 
                    chains_memmap_path, atom_names_memmap_path]:
            if not os.path.exists(path):
                print(f"RAY_WORKER: Error: Memmap file not found: {path}")
                return None, None, None
            
        # Open memmap files with error handling
        try:
            protein_all_coords_np = np.memmap(coords_memmap_path, dtype=np.float32, mode='r', shape=full_coords_shape)
            protein_all_elements_np = np.memmap(elements_memmap_path, dtype=np.int32, mode='r', shape=full_elements_shape)
            all_residue_nums_np = np.memmap(residues_memmap_path, dtype=np.int64, mode='r', shape=full_residues_shape)
            all_chain_ids_np = np.memmap(chains_memmap_path, dtype=np.int32, mode='r', shape=full_chains_shape)
            all_atom_names_np = np.memmap(atom_names_memmap_path, dtype=np.int32, mode='r', shape=full_atom_names_shape)
        except FileNotFoundError as fnf:
            print(f"RAY_WORKER: FileNotFoundError accessing memmap in Ray worker: {fnf}")
            return None, None, None
        except ValueError as ve:
            print(f"RAY_WORKER: ValueError accessing memmap in Ray worker (shape/dtype mismatch?): {ve}")
            return None, None, None
        except Exception as e:
            print(f"RAY_WORKER: Unexpected error accessing memmap in Ray worker: {e}")
            return None, None, None
            
        # Validate array consistency
        if protein_all_coords_np.shape[0] != protein_all_elements_np.shape[0]:
            print(f"RAY_WORKER: Array shape mismatch coords:{protein_all_coords_np.shape[0]} vs elements:{protein_all_elements_np.shape[0]}")
            return None, None, None
            
        if protein_all_coords_np.shape[0] == 0:
            print(f"RAY_WORKER: Empty coordinate array")
            return None, None, None
        
        # Build KDTree locally
        try:
            kdtree = build_kdtree_fn(protein_all_coords_np)
        except Exception as e_kdtree:
            print(f"RAY_WORKER: Error building KDTree: {e_kdtree}")
            return None, None, None
        
        _protein_id, chain_id_target, resnum_target, _resname_letter, _bfactor = residue_key_as_task
        
        # Create a mask for the specific target residue
        # Ensure resnum_target is the correct type for comparison with all_residue_nums_np (which is int64)
        if not isinstance(resnum_target, (int, np.integer)):
            try:
                resnum_target_int = int(resnum_target)
            except ValueError:
                print(f"RAY_WORKER: Error: resnum_target '{resnum_target}' is not a valid integer for {residue_key_as_task}")
                return None, None, None # Cannot proceed
        else:
            resnum_target_int = resnum_target

        # Get chain code from original chain ID
        try:
            # Find chain code that maps to chain_id_target
            chain_code = None
            for code, chain in chain_lookup.items():
                if chain == chain_id_target:
                    chain_code = code
                    break
                    
            if chain_code is None:
                print(f"RAY_WORKER: Could not find chain code for '{chain_id_target}' in lookup table")
                return None, None, None
        except Exception as e_chain:
            print(f"RAY_WORKER: Error finding chain code: {e_chain}")
            return None, None, None
            
        # Safely apply mask with bounds checking
        try:
            residue_mask = (all_chain_ids_np == chain_code) & (all_residue_nums_np == resnum_target_int)
        except Exception as e_mask:
            print(f"RAY_WORKER: Error creating residue mask: {e_mask}")
            return None, None, None
        
        if not np.any(residue_mask):
            # This means no atoms were found for this specific residue_key in the memmapped arrays.
            # This could happen if residue_keys_for_tasks was derived from a slightly different atom_df
            # than the one used to create memmaps, or if the residue is genuinely empty/filtered out.
            # print(f"RAY_WORKER: Warning: No atoms found for residue {chain_id_target}-{resnum_target_int} via memmap mask.")
            return None, None, None 

        # Extract coordinates and atom names for only the atoms of the target residue
        try:
            target_residue_coords_np = protein_all_coords_np[residue_mask]
            target_residue_atom_names_codes = all_atom_names_np[residue_mask]
            
            # Convert atom name codes back to strings
            target_residue_atom_names = np.array([atom_name_lookup.get(code, "UNK") for code in target_residue_atom_names_codes])
        except Exception as e_extract:
            print(f"RAY_WORKER: Error extracting data with mask: {e_extract}")
            return None, None, None

        if target_residue_coords_np.shape[0] == 0:
            # This means the mask matched, but resulted in zero atoms (e.g. if protein_all_coords_np was shorter than expected)
            # Unlikely if np.any(residue_mask) was true and arrays are consistent, but defensive.
            # print(f"RAY_WORKER: Warning: Zero atoms extracted for residue {chain_id_target}-{resnum_target_int} despite mask match.")
            return None, None, None
            
        # Make sure lengths match
        if len(target_residue_coords_np) != len(target_residue_atom_names):
            print(f"RAY_WORKER: Length mismatch in extracted data: coords={len(target_residue_coords_np)}, names={len(target_residue_atom_names)}")
            return None, None, None

        # Construct the minimal res_df required by sample_functional_center
        # This DataFrame contains *only* atoms of the target residue.
        try:
            res_df_data = {
                'x': target_residue_coords_np[:, 0].astype(np.float32),
                'y': target_residue_coords_np[:, 1].astype(np.float32),
                'z': target_residue_coords_np[:, 2].astype(np.float32),
                'name': target_residue_atom_names,
                # Add 'residue' column as it's used by _process_residue_env_worker's own filtering step.
                # It will select all rows from this already-filtered df.
                'residue': np.full(len(target_residue_atom_names), resnum_target_int, dtype=np.int64),
                'chain': np.full(len(target_residue_atom_names), chain_id_target)
            }
            res_df_for_single_residue = pd.DataFrame(res_df_data)
        except Exception as e_df:
            print(f"RAY_WORKER: Error creating DataFrame for residue: {e_df}")
            return None, None, None

        # Convert element codes back to strings for the whole protein
        try:
            # Use numpy advanced indexing just for the environment
            # Only prepare this when needed for KDTree neighbors
            # element_strings = np.array([element_lookup.get(code, "X") for code in protein_all_elements_np])
            # This array prep is done lazily in the _process_residue_env_worker when it gets neighbors
            elements_fn = lambda code: element_lookup.get(code, "X")
        except Exception as e_elements:
            print(f"RAY_WORKER: Error preparing element strings: {e_elements}")
            return None, None, None

        # _process_residue_env_worker expects a dictionary {chain_id: chain_df}
        # We provide our highly specific res_df_for_single_residue as the chain_df for the target chain.
        mock_chain_atom_dfs_global = {chain_id_target: res_df_for_single_residue}

        try:
            # Wrapper for _process_residue_env_worker that modifies the protein_all_elements_np parameter
            def modified_worker(residue_key, protein_all_coords_np, protein_all_elements_np, 
                              chain_atom_dfs_global, kdtree_global, env_radius):
                """
                Wraps _process_residue_env_worker to handle element code conversion just-in-time.
                """
                # Get indices from KDTree for environment
                center_coords = sample_functional_center(
                    chain_atom_dfs_global[residue_key[1]], 
                    (residue_key[3], int(residue_key[2])), 
                    train_mode=False
                )
                
                if center_coords is None:
                    print(f"RAY_WORKER: Could not calculate center for {residue_key}")
                    return None, None
                    
                # Get environment atoms
                pt_indices = kdtree_global.query_ball_point(center_coords, r=env_radius)
                
                if not pt_indices or len(pt_indices) == 0:
                    print(f"RAY_WORKER: No environment atoms found for {residue_key}")
                    return None, None
                
                # Convert element codes to strings just for the environment atoms
                env_element_codes = protein_all_elements_np[pt_indices]
                env_element_strings = np.array([elements_fn(code) for code in env_element_codes])
                
                # Create metadata
                metadata = {
                    'protein_id': residue_key[0],
                    'chain': residue_key[1],
                    'resid': f"{residue_key[3]}{residue_key[2]}",
                    'resname_letter': residue_key[3],
                    'resnum': residue_key[2],
                    'confidence': residue_key[4]
                }
                
                # Extract coordinates for the environment
                env_coords_np = protein_all_coords_np[pt_indices]
                
                return metadata, (env_coords_np, env_element_strings)
                
            metadata, env_data = modified_worker(
                residue_key_as_task,
                protein_all_coords_np, 
                protein_all_elements_np,
                mock_chain_atom_dfs_global,
                kdtree,
                env_radius_val
            )
        except Exception as e_worker:
            print(f"RAY_WORKER: Error in worker: {e_worker}")
            traceback.print_exc()
            return None, None, None
            
        if metadata is None: # Worker failed
            return None, None, None
        
        # Make sure we don't return None for env_data components
        if env_data is None or len(env_data) < 2 or env_data[0] is None or env_data[1] is None:
            print(f"RAY_WORKER: Worker returned incomplete env_data")
            return None, None, None
            
        # Clean up resources explicitly (to be extra safe)
        try:
            del protein_all_coords_np, protein_all_elements_np, all_residue_nums_np, all_chain_ids_np, all_atom_names_np
            del kdtree
        except:
            pass
            
        return metadata, env_data[0], env_data[1] # metadata, env_coords_np, env_elements_np
        
    except Exception as e_global:
        print(f"RAY_WORKER: Unhandled exception in process_residue_for_ray: {e_global}")
        traceback.print_exc()
        return None, None, None

@ray.remote
class RayEmbeddingActor:
    def __init__(self,
                 coords_memmap_path_actor, elements_memmap_path_actor,
                 residues_memmap_path_actor, chains_memmap_path_actor,
                 atom_names_memmap_path_actor,
                 element_dict_path_actor, chain_dict_path_actor, atom_name_dict_path_actor,
                 full_coords_shape_actor, full_elements_shape_actor,
                 full_residues_shape_actor, full_chains_shape_actor,
                 full_atom_names_shape_actor,
                 element_dict_shape_actor, chain_dict_shape_actor, atom_name_dict_shape_actor,
                 env_radius_actor):
        # Store memmap paths and shapes
        self.coords_memmap_path = coords_memmap_path_actor
        self.elements_memmap_path = elements_memmap_path_actor
        self.residues_memmap_path = residues_memmap_path_actor
        self.chains_memmap_path = chains_memmap_path_actor
        self.atom_names_memmap_path = atom_names_memmap_path_actor
        
        # Dictionary lookup paths
        self.element_dict_path = element_dict_path_actor
        self.chain_dict_path = chain_dict_path_actor
        self.atom_name_dict_path = atom_name_dict_path_actor

        # Shapes
        self.full_coords_shape = full_coords_shape_actor
        self.full_elements_shape = full_elements_shape_actor
        self.full_residues_shape = full_residues_shape_actor
        self.full_chains_shape = full_chains_shape_actor
        self.full_atom_names_shape = full_atom_names_shape_actor
        
        # Dictionary shapes
        self.element_dict_shape = element_dict_shape_actor
        self.chain_dict_shape = chain_dict_shape_actor
        self.atom_name_dict_shape = atom_name_dict_shape_actor
        
        # Store environment radius
        self.env_radius = env_radius_actor
        
        # Initialize dictionaries
        self._element_lookup = None
        self._chain_lookup = None
        self._atom_name_lookup = None
        
        # Preload dictionaries for faster lookups
        try:
            self._init_lookup_tables()
        except Exception as e:
            print(f"Warning: Failed to initialize lookup tables in Ray actor: {e}")
    
    def _init_lookup_tables(self):
        """Initialize code-to-string lookup dictionaries from memmap files"""
        # Load element dictionary
        element_dict = np.memmap(self.element_dict_path, dtype=[('code', np.int32), ('element', 'U4')], 
                                 mode='r', shape=self.element_dict_shape)
        self._element_lookup = {code: elem for code, elem in element_dict}
        
        # Load chain dictionary
        chain_dict = np.memmap(self.chain_dict_path, dtype=[('code', np.int32), ('chain', 'U4')], 
                               mode='r', shape=self.chain_dict_shape)
        self._chain_lookup = {code: chain for code, chain in chain_dict}
        
        # Load atom name dictionary
        atom_name_dict = np.memmap(self.atom_name_dict_path, dtype=[('code', np.int32), ('name', 'U4')], 
                                   mode='r', shape=self.atom_name_dict_shape)
        self._atom_name_lookup = {code: name for code, name in atom_name_dict}

    def _build_kdtree(self, coords_np):
        """Builds a KDTree from coordinate data"""
        if coords_np.shape[0] == 0:
            raise ValueError("Cannot build KDTree with empty coordinates")
        return scipy.spatial.cKDTree(coords_np, compact_nodes=True, copy_data=False)

    def process_residue_environment(self, residue_key_task):
        # residue_key_task is: (protein_id, chain_id, resnum, resname_letter, bfactor)
        return process_residue_for_ray(
            residue_key_task,
            self.coords_memmap_path, self.elements_memmap_path,
            self.residues_memmap_path, self.chains_memmap_path, self.atom_names_memmap_path,
            self.full_coords_shape, self.full_elements_shape,
            self.full_residues_shape, self.full_chains_shape, self.full_atom_names_shape,
            self._element_lookup, self._chain_lookup, self._atom_name_lookup,
            self.env_radius, # from self
            self._build_kdtree # Pass the method itself
        )

# %% [markdown]
# ## Main Embedding Generation Function (Refactored for Ray)
# %%
def generate_single_pdb_embedding_ray(atom_df_input: pd.DataFrame, protein_id_input: str,
                                   model_main_process, device_main_process, # model and device for main process
                                   include_hets=INCLUDE_HETS,
                                   env_radius=ENV_RADIUS,
                                   edge_cutoff=EDGE_CUTOFF,
                                   num_rbf=NUM_RBF,
                                   num_workers=NUM_WORKERS): # Used for Ray actors
    overall_start_time = time.perf_counter()
    print("-" * 30)
    print(f"Starting Ray embedding generation for: {protein_id_input}")
    print(f"Parameters: EnvRadius={env_radius}, EdgeCutoff={edge_cutoff}, IncludeHets={include_hets}, RayWorkers={num_workers}")
    print("-" * 30)

    # --- 1. Load and Preprocess PDB (Main Process) ---
    # This part remains largely the same
    current_step_start_time = time.perf_counter()
    atom_df = None
    try:
        atom_df = atom_df_input.copy()
        if atom_df is None or atom_df.empty:
            raise ValueError(f"Input atom_df_input is None or empty for {protein_id_input}")

        atom_df = first_model_filter(atom_df)
        atom_df = atom_df[~atom_df.hetero.str.contains('W', na=False)]
        atom_df = atom_df[atom_df['element'] != 'H']
        if not include_hets:
            if hasattr(atom_info, 'aa') and isinstance(atom_info.aa, (list, set)):
                if 'resname' in atom_df.columns:
                    atom_df = atom_df[atom_df.resname.isin(atom_info.aa)]
                else:
                    print(f"Warning: 'resname' column missing, cannot filter hets for {protein_id_input}.")
            else:
                print("Warning: atom_info.aa not found, cannot filter hets.")
        atom_df = atom_df.reset_index(drop=True)
        if atom_df.empty: raise ValueError(f"PDB empty after filtering: {protein_id_input}")
        
        essential_cols = {'resname', 'chain', 'residue', 'element', 'x', 'y', 'z', 'name', 'bfactor'}
        missing_cols = essential_cols - set(atom_df.columns)
        if missing_cols: raise ValueError(f"Missing essential columns: {missing_cols}")
        atom_df['id'] = protein_id_input
        atom_df['residue'] = pd.to_numeric(atom_df['residue']) # Ensure residue is numeric
        print(f"Step 1: Loaded and preprocessed. Atoms: {len(atom_df)}. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 1: {e}"); traceback.print_exc(); return None, None, time.perf_counter() - overall_start_time

    # --- 2. Build KDTree (Main Process) ---
    kdtree = None
    current_step_start_time = time.perf_counter()
    try:
        coords_for_kdtree = atom_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        if coords_for_kdtree.shape[0] == 0: raise ValueError("No coords for KDTree.")
        kdtree = scipy.spatial.cKDTree(coords_for_kdtree, compact_nodes=True, copy_data=False)
        print(f"Step 2: Built KDTree for {kdtree.n} atoms. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 2: {e}"); traceback.print_exc(); return None, None, time.perf_counter() - overall_start_time

    # --- 3. Identify Residues (Main Process) ---
    residue_keys_for_tasks = []
    current_step_start_time = time.perf_counter()
    try:
        if atom_info.aa_to_letter_dict:
            atom_df['resname_letter'] = atom_df['resname'].map(atom_info.aa_to_letter_dict)
        else:
            atom_df['resname_letter'] = atom_df['resname'].apply(atom_info.aa_to_letter)
        
        residue_info_df = atom_df[['chain', 'residue', 'resname_letter', 'bfactor']].drop_duplicates(subset=['chain', 'residue'])
        standard_letters = set(atom_info.aa_abbr) - {'X'} if hasattr(atom_info, 'aa_abbr') else set('ACDEFGHIKLMNPQRSTVWY')
        residue_info_df = residue_info_df[residue_info_df['resname_letter'].isin(standard_letters) & residue_info_df['resname_letter'].notna()]
        if residue_info_df.empty: raise ValueError(f"No standard AA residues found in {protein_id_input}.")

        residue_keys_for_tasks = [
            (protein_id_input, row['chain'], row['residue'], row['resname_letter'], row['bfactor'])
            for _, row in residue_info_df.iterrows()
        ]
        print(f"Step 3: Identified {len(residue_keys_for_tasks)} standard residues. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 3: {e}"); traceback.print_exc(); return None, None, time.perf_counter() - overall_start_time

    # --- 4. Prepare Memory-Mapped Files (Main Process) ---
    print("Step 4a: Preparing memory-mapped files...")
    current_step_start_time = time.perf_counter()
    
    # Convert object arrays to numeric codes to avoid serialization issues
    # Elements conversion
    unique_elements = atom_df['element'].unique()
    element_to_code = {elem: i for i, elem in enumerate(unique_elements)}
    atom_df['element_code'] = atom_df['element'].map(element_to_code)
    
    # Chain ID conversion
    unique_chains = atom_df['chain'].unique()
    chain_to_code = {chain: i for i, chain in enumerate(unique_chains)}
    atom_df['chain_code'] = atom_df['chain'].map(chain_to_code)
    
    # Atom name conversion
    unique_atom_names = atom_df['name'].unique()
    atom_name_to_code = {name: i for i, name in enumerate(unique_atom_names)}
    atom_df['name_code'] = atom_df['name'].map(atom_name_to_code)
    
    # Data for memmap - use numeric types where possible
    # Ensure arrays are C-contiguous by using numpy.ascontiguousarray()
    all_coords_np_orig = np.ascontiguousarray(atom_df[['x', 'y', 'z']].to_numpy(dtype=np.float32))
    all_elements_np_orig = np.ascontiguousarray(atom_df['element_code'].to_numpy(dtype=np.int32))
    all_residues_np_orig = np.ascontiguousarray(atom_df['residue'].to_numpy(dtype=np.int64))
    all_chains_np_orig = np.ascontiguousarray(atom_df['chain_code'].to_numpy(dtype=np.int32))
    all_atom_names_np_orig = np.ascontiguousarray(atom_df['name_code'].to_numpy(dtype=np.int32))
    
    # Validate memory layout
    assert all_coords_np_orig.flags['C_CONTIGUOUS'], "Coords array not C-contiguous"
    assert all_elements_np_orig.flags['C_CONTIGUOUS'], "Elements array not C-contiguous"
    assert all_residues_np_orig.flags['C_CONTIGUOUS'], "Residues array not C-contiguous"
    assert all_chains_np_orig.flags['C_CONTIGUOUS'], "Chains array not C-contiguous"
    assert all_atom_names_np_orig.flags['C_CONTIGUOUS'], "Atom names array not C-contiguous"
    
    # Create additional memmap files for lookup dictionaries
    element_dict_np = np.ascontiguousarray(np.array([(i, elem) for elem, i in element_to_code.items()], 
                               dtype=[('code', np.int32), ('element', 'U4')]))
    chain_dict_np = np.ascontiguousarray(np.array([(i, chain) for chain, i in chain_to_code.items()], 
                             dtype=[('code', np.int32), ('chain', 'U4')]))
    atom_name_dict_np = np.ascontiguousarray(np.array([(i, name) for name, i in atom_name_to_code.items()], 
                                 dtype=[('code', np.int32), ('name', 'U4')]))

    # Shapes for memmap
    full_coords_shape = all_coords_np_orig.shape
    full_elements_shape = all_elements_np_orig.shape
    full_residues_shape = all_residues_np_orig.shape
    full_chains_shape = all_chains_np_orig.shape
    full_atom_names_shape = all_atom_names_np_orig.shape
    
    # Element dictionary lookup shapes
    element_dict_shape = element_dict_np.shape
    chain_dict_shape = chain_dict_np.shape  
    atom_name_dict_shape = atom_name_dict_np.shape

    # Clean up old memmap files if they exist
    for p in [MEMMAP_COORDS_PATH, MEMMAP_ELEMENTS_PATH, MEMMAP_RESIDUES_PATH, 
              MEMMAP_CHAINS_PATH, MEMMAP_ATOM_NAMES_PATH, 
              MEMMAP_ELEMENT_DICT_PATH, MEMMAP_CHAIN_DICT_PATH, MEMMAP_ATOM_NAME_DICT_PATH]:
        if os.path.exists(p): os.remove(p)

    # Create and write to memmap files
    mmap_coords = np.memmap(MEMMAP_COORDS_PATH, dtype=np.float32, mode='w+', shape=full_coords_shape)
    mmap_coords[:] = all_coords_np_orig[:]
    mmap_coords.flush(); del mmap_coords # Flush and delete handle
    
    mmap_elements = np.memmap(MEMMAP_ELEMENTS_PATH, dtype=np.int32, mode='w+', shape=full_elements_shape)
    mmap_elements[:] = all_elements_np_orig[:]
    mmap_elements.flush(); del mmap_elements

    mmap_residues = np.memmap(MEMMAP_RESIDUES_PATH, dtype=np.int64, mode='w+', shape=full_residues_shape)
    mmap_residues[:] = all_residues_np_orig[:]
    mmap_residues.flush(); del mmap_residues

    mmap_chains = np.memmap(MEMMAP_CHAINS_PATH, dtype=np.int32, mode='w+', shape=full_chains_shape)
    mmap_chains[:] = all_chains_np_orig[:]
    mmap_chains.flush(); del mmap_chains

    mmap_atom_names = np.memmap(MEMMAP_ATOM_NAMES_PATH, dtype=np.int32, mode='w+', shape=full_atom_names_shape)
    mmap_atom_names[:] = all_atom_names_np_orig[:]
    mmap_atom_names.flush(); del mmap_atom_names
    
    # Write dictionaries to memmap files for lookup
    mmap_element_dict = np.memmap(MEMMAP_ELEMENT_DICT_PATH, dtype=element_dict_np.dtype, mode='w+', shape=element_dict_shape)
    mmap_element_dict[:] = element_dict_np[:]
    mmap_element_dict.flush(); del mmap_element_dict
    
    mmap_chain_dict = np.memmap(MEMMAP_CHAIN_DICT_PATH, dtype=chain_dict_np.dtype, mode='w+', shape=chain_dict_shape)
    mmap_chain_dict[:] = chain_dict_np[:]
    mmap_chain_dict.flush(); del mmap_chain_dict
    
    mmap_atom_name_dict = np.memmap(MEMMAP_ATOM_NAME_DICT_PATH, dtype=atom_name_dict_np.dtype, mode='w+', shape=atom_name_dict_shape)
    mmap_atom_name_dict[:] = atom_name_dict_np[:]
    mmap_atom_name_dict.flush(); del mmap_atom_name_dict
    
    del all_coords_np_orig, all_elements_np_orig, all_residues_np_orig, all_chains_np_orig, all_atom_names_np_orig
    del element_dict_np, chain_dict_np, atom_name_dict_np
    # Store mapping dictionaries for later use
    memmap_lookup = {
        'element_to_code': element_to_code,
        'chain_to_code': chain_to_code,
        'atom_name_to_code': atom_name_to_code
    }

    print(f"Step 4a: Memory-mapped files prepared. Time: {time.perf_counter() - current_step_start_time:.4f}s")

    # --- 4b. Initialize Ray and Actors ---
    current_step_start_time = time.perf_counter()
    if not ray.is_initialized():
        # Initialize Ray with additional configuration options
        ray_init_options = {
            "num_cpus": num_workers,
            "ignore_reinit_error": True,
            # Add runtime_env to share directory access permissions
            "runtime_env": {
                "env_vars": {
                    "RAY_PICKLE_VERBOSE_DEBUG": "1",  # For better debugging
                }
            },
            # Configure object memory store limits
            "_memory": int(0.8 * psutil.virtual_memory().total),  # Limit to 80% of system memory
            "local_mode": False,  # Set to True for debugging/testing
        }
        
        ray.init(**ray_init_options)
        print(f"Ray initialized with options: {ray_init_options}")
    else:
        print("Ray already initialized.")

    # Create Ray actors - pass dictionaries instead of kdtree
    actors = [RayEmbeddingActor.remote(MEMMAP_COORDS_PATH, MEMMAP_ELEMENTS_PATH,
                                     MEMMAP_RESIDUES_PATH, MEMMAP_CHAINS_PATH, MEMMAP_ATOM_NAMES_PATH,
                                     MEMMAP_ELEMENT_DICT_PATH, MEMMAP_CHAIN_DICT_PATH, MEMMAP_ATOM_NAME_DICT_PATH,
                                     full_coords_shape, full_elements_shape,
                                     full_residues_shape, full_chains_shape, full_atom_names_shape,
                                     element_dict_shape, chain_dict_shape, atom_name_dict_shape,
                                     env_radius)
              for _ in range(num_workers)]
    print(f"Step 4b: Ray actors created. Num actors: {len(actors)}. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    
    # --- 4c. Dispatch tasks to Ray Actors ---
    print(f"Step 4c: Dispatching {len(residue_keys_for_tasks)} tasks to Ray actors...")
    current_step_start_time = time.perf_counter()
    results_refs = []
    # Simple round-robin dispatch
    actor_idx = 0
    for res_key_task in residue_keys_for_tasks:
        actor = actors[actor_idx % num_workers]
        results_refs.append(actor.process_residue_environment.remote(res_key_task))
        actor_idx += 1
    
    print(f"Step 4c: Tasks dispatched. Time: {time.perf_counter() - current_step_start_time:.4f}s")

    # --- 4d. Collect results from Ray ---
    print(f"Step 4d: Collecting results from Ray (timeout = 60s per batch of gets, or total)...")
    current_step_start_time = time.perf_counter()
    
    # Get results with a timeout. This is a simple way.
    # For very long task lists, might need to get in batches.
    try:
        # Using ray.get with a list of ObjectRefs
        raw_results = ray.get(results_refs, timeout=60.0) # Timeout for all results
    except ray.exceptions.GetTimeoutError:
        print("Warning: Timeout occurred while getting results from Ray actors.")
        # Attempt to salvage what we can
        ready_refs, remaining_refs = ray.wait(results_refs, num_returns=len(results_refs), timeout=1.0) # Quick check
        raw_results = ray.get(ready_refs) # Get what's ready
        print(f"  Retrieved {len(raw_results)} results before timeout forced collection.")
        # You might want to cancel remaining_refs if Ray version supports it easily
        # for ref in remaining_refs: ray.cancel(ref, force=True) # Requires Ray 1.9+
    except Exception as e_get:
        print(f"Error getting results from Ray: {e_get}"); traceback.print_exc()
        raw_results = []


    cpu_pool_results = [res for res in raw_results if res is not None and res[0] is not None]
    del raw_results # Free memory
    print(f"Step 4d: Ray results collected. Valid results: {len(cpu_pool_results)}. Time: {time.perf_counter() - current_step_start_time:.4f}s")

    # --- Cleanup memmap files and KDTree related data earlier if possible ---
    # kdtree is used by actors, atom_df is used for memmaps.
    # del kdtree # if kdtree_ref was used, this would be fine. Here it's copied to actors.
    del atom_df # Original dataframe no longer needed after memmaps and residue_keys

    # --- 4.5 Create GPU Graphs in Main Process (remains similar) ---
    if not cpu_pool_results:
        print("Error: No environments identified by Ray workers.")
        # Clean up memmap files before exiting -- REMOVED, handled by atexit now
        return None, None, time.perf_counter() - overall_start_time

    print(f"Step 4.5: Creating {len(cpu_pool_results)} PyG graphs on GPU ({device_main_process})...")
    current_step_start_time = time.perf_counter()
    gpu_graphs_list = []
    final_metadata_list = []

    for result_item in cpu_pool_results:
        # Expected item: (metadata, env_coords_np, env_elements_np)
        metadata, env_coords_np_worker, env_elements_np_worker = result_item
        if metadata is None or env_coords_np_worker is None or env_elements_np_worker is None:
            # print(f"Warning: Skipping None result from worker for {metadata.get('resid') if metadata else 'Unknown'}")
            continue
        
        gpu_graph = create_pyg_graph_on_gpu(env_coords_np_worker, env_elements_np_worker,
                                            device_main_process, edge_cutoff, num_rbf)
        if gpu_graph:
            gpu_graphs_list.append(gpu_graph)
            final_metadata_list.append(metadata)

    if not gpu_graphs_list:
        print("Error: Failed to create any graphs on GPU from Ray worker results.")
        # for p in [MEMMAP_COORDS_PATH, MEMMAP_ELEMENTS_PATH, MEMMAP_RESIDUES_PATH, MEMMAP_CHAINS_PATH, MEMMAP_ATOM_NAMES_PATH]:
        #     if os.path.exists(p): os.remove(p)
        return None, (final_metadata_list if final_metadata_list else None), time.perf_counter() - overall_start_time
    
    print(f"Step 4.5: Created {len(gpu_graphs_list)} graphs on GPU. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    metadata_list = final_metadata_list

    # --- 5. Batch Graphs (remains similar) ---
    graph_batch_gpu = None
    current_step_start_time = time.perf_counter()
    try:
        graph_batch_gpu = Batch.from_data_list(gpu_graphs_list)
        graph_batch_gpu = graph_batch_gpu.to(device_main_process)
        print(f"Step 5: Batching complete. Batch is on device: {graph_batch_gpu.x.device}. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 5: {e}"); traceback.print_exc()
        # for p in [MEMMAP_COORDS_PATH, MEMMAP_ELEMENTS_PATH, MEMMAP_RESIDUES_PATH, MEMMAP_CHAINS_PATH, MEMMAP_ATOM_NAMES_PATH]:
        #     if os.path.exists(p): os.remove(p)
        return None, metadata_list, time.perf_counter() - overall_start_time
    finally:
        if gpu_graphs_list: del gpu_graphs_list

    # --- 6. Run Model Inference (remains similar) ---
    print(f"Step 6: Running inference on device {device_main_process}...")
    current_step_start_time = time.perf_counter()
    embeddings_np = None
    try:
        model_setup_time = time.perf_counter()
        print(f"  Moving model to {device_main_process} (if not already there)...") # Model should be on device already from run_pipeline
        model_main_process.to(device_main_process) # Ensure it is
        print(f"  Model on device. Time: {time.perf_counter() - model_setup_time:.4f}s")

        if COMPILE_MODEL and hasattr(torch, 'compile') and not isinstance(model_main_process, torch.jit.ScriptModule) and not model_main_process.__class__.__name__.endswith("CompiledModule"): # Avoid re-compiling
            # Check if model is already compiled might be tricky. This is a basic check.
            compile_start_time = time.perf_counter()
            print("  Compiling model on target device (this may take a moment)...")
            try:
                model_main_process = torch.compile(model_main_process, mode="reduce-overhead")
                print(f"  Model compiled successfully on target device. Time: {time.perf_counter() - compile_start_time:.4f}s")
            except Exception as e_compile:
                print(f"  Warning: Model compilation on {device_main_process} failed: {e_compile}")
        
        actual_inference_start_time = time.perf_counter()
        model_main_process.eval()
        with torch.no_grad():
            with torch.autocast(device_type=str(device_main_process.type), dtype=torch.float16, enabled=(str(device_main_process.type) == 'cuda')):
                embs, _ = model_main_process.online_encoder(graph_batch_gpu, return_projection=False)
                embeddings_np = embs.float().cpu().numpy()
        
        print(f"  Actual inference. Time: {time.perf_counter() - actual_inference_start_time:.4f}s")
        print(f"Step 6: Inference finished. Output shape: {embeddings_np.shape}. Total Step 6 Time: {time.perf_counter() - current_step_start_time:.4f}s")
        if embeddings_np.shape[0] != len(metadata_list):
            raise ValueError(f"Embedding count mismatch: Expected {len(metadata_list)}, Got {embeddings_np.shape[0]}.")
    except RuntimeError as e_rt:
        if "CUDA out of memory" in str(e_rt): print("CUDA OOM during inference!")
        else: print(f"Runtime error during inference: {e_rt}")
        traceback.print_exc()
        return None, metadata_list, time.perf_counter() - overall_start_time # Memmap files cleaned in finally
    except Exception as e_inf:
        print(f"Error in Step 6 (Inference): {e_inf}"); traceback.print_exc()
        return None, metadata_list, time.perf_counter() - overall_start_time # Memmap files cleaned in finally
    finally:
        if graph_batch_gpu is not None: del graph_batch_gpu
        if 'embs' in locals(): del embs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Cleanup memmap files -- REMOVED, handled by atexit now
        # for p in [MEMMAP_COORDS_PATH, MEMMAP_ELEMENTS_PATH, MEMMAP_RESIDUES_PATH, MEMMAP_CHAINS_PATH, MEMMAP_ATOM_NAMES_PATH]:
        #     if os.path.exists(p): os.remove(p)
        # print("Memmap files cleaned up.")


    # --- 7. Return Results ---
    total_time_taken = time.perf_counter() - overall_start_time
    print("-" * 30)
    print(f"Pipeline finished successfully for {protein_id_input}.")
    print(f"Total processing time (perf_counter): {total_time_taken:.4f} seconds.")
    print("-" * 30)
    return embeddings_np, metadata_list, total_time_taken

def generate_single_pdb_embedding_optimized(atom_df_input: pd.DataFrame, protein_id_input: str,
                                  model_main_process, device_main_process,
                                  include_hets=INCLUDE_HETS,
                                  env_radius=ENV_RADIUS,
                                  edge_cutoff=EDGE_CUTOFF,
                                  num_rbf=NUM_RBF,
                                  num_workers=NUM_WORKERS,
                                  use_ray=False):
    """
    Optimized embedding generation that chooses between multiprocessing and Ray
    based on system configuration and worker count.
    
    Args:
        atom_df_input: DataFrame with protein atom data
        protein_id_input: Protein identifier
        model_main_process: Model instance for inference
        device_main_process: Target device for inference
        include_hets: Whether to include heteroatoms
        env_radius: Radius for residue environment
        edge_cutoff: Edge cutoff for graph creation
        num_rbf: Number of radial basis functions
        num_workers: Number of workers to use
        use_ray: Force use of Ray if True, otherwise auto-select
        
    Returns:
        tuple: (embeddings, metadata, duration)
    """
    # Choose implementation based on system configuration
    if not use_ray and num_workers <= os.cpu_count():
        return generate_single_pdb_embedding_threads(
            atom_df_input, protein_id_input, model_main_process, device_main_process,
            include_hets, env_radius, edge_cutoff, num_rbf, num_workers
        )
    else:
        return generate_single_pdb_embedding_ray(
            atom_df_input, protein_id_input, model_main_process, device_main_process,
            include_hets, env_radius, edge_cutoff, num_rbf, num_workers
        )

def generate_single_pdb_embedding_threads(atom_df_input: pd.DataFrame, protein_id_input: str,
                                    model_main_process, device_main_process,
                                    include_hets=INCLUDE_HETS,
                                    env_radius=ENV_RADIUS,
                                    edge_cutoff=EDGE_CUTOFF,
                                    num_rbf=NUM_RBF,
                                    num_workers=NUM_WORKERS):
    """
    Generate embeddings using optimized multiprocessing with shared memory.
    Incorporates memory safety fixes but uses efficient thread-based parallelism.
    """
    overall_start_time = time.perf_counter()
    print("-" * 30)
    print(f"Starting OPTIMIZED thread-based embedding generation for: {protein_id_input}")
    print(f"Parameters: EnvRadius={env_radius}, EdgeCutoff={edge_cutoff}, IncludeHets={include_hets}, Workers={num_workers}")
    print("-" * 30)

    # --- 1. Load and Preprocess PDB (Main Process) ---
    current_step_start_time = time.perf_counter()
    atom_df = None
    try:
        atom_df = atom_df_input.copy()
        if atom_df is None or atom_df.empty:
            raise ValueError(f"Input atom_df_input is None or empty for {protein_id_input}")

        atom_df = first_model_filter(atom_df)
        atom_df = atom_df[~atom_df.hetero.str.contains('W', na=False)]
        atom_df = atom_df[atom_df['element'] != 'H']
        if not include_hets:
            if hasattr(atom_info, 'aa') and isinstance(atom_info.aa, (list, set)):
                if 'resname' in atom_df.columns:
                    atom_df = atom_df[atom_df.resname.isin(atom_info.aa)]
                else:
                    print(f"Warning: 'resname' column missing, cannot filter hets for {protein_id_input}.")
            else:
                print("Warning: atom_info.aa not found, cannot filter hets.")
        atom_df = atom_df.reset_index(drop=True)
        if atom_df.empty: raise ValueError(f"PDB empty after filtering: {protein_id_input}")
        
        essential_cols = {'resname', 'chain', 'residue', 'element', 'x', 'y', 'z', 'name', 'bfactor'}
        missing_cols = essential_cols - set(atom_df.columns)
        if missing_cols: raise ValueError(f"Missing essential columns: {missing_cols}")
        atom_df['id'] = protein_id_input
        atom_df['residue'] = pd.to_numeric(atom_df['residue']) # Ensure residue is numeric
        print(f"Step 1: Loaded and preprocessed. Atoms: {len(atom_df)}. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 1: {e}"); traceback.print_exc(); return None, None, time.perf_counter() - overall_start_time

    # --- 2. Build KDTree (Main Process) ---
    kdtree = None
    current_step_start_time = time.perf_counter()
    try:
        coords_for_kdtree = np.ascontiguousarray(atom_df[['x', 'y', 'z']].to_numpy(dtype=np.float32))
        if coords_for_kdtree.shape[0] == 0: raise ValueError("No coords for KDTree.")
        kdtree = scipy.spatial.cKDTree(coords_for_kdtree, compact_nodes=True, copy_data=False)
        print(f"Step 2: Built KDTree for {kdtree.n} atoms. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 2: {e}"); traceback.print_exc(); return None, None, time.perf_counter() - overall_start_time

    # --- 3. Identify Residues (Main Process) ---
    residue_keys_for_tasks = []
    current_step_start_time = time.perf_counter()
    try:
        if atom_info.aa_to_letter_dict:
            atom_df['resname_letter'] = atom_df['resname'].map(atom_info.aa_to_letter_dict)
        else:
            atom_df['resname_letter'] = atom_df['resname'].apply(atom_info.aa_to_letter)
        
        residue_info_df = atom_df[['chain', 'residue', 'resname_letter', 'bfactor']].drop_duplicates(subset=['chain', 'residue'])
        standard_letters = set(atom_info.aa_abbr) - {'X'} if hasattr(atom_info, 'aa_abbr') else set('ACDEFGHIKLMNPQRSTVWY')
        residue_info_df = residue_info_df[residue_info_df['resname_letter'].isin(standard_letters) & residue_info_df['resname_letter'].notna()]
        if residue_info_df.empty: raise ValueError(f"No standard AA residues found in {protein_id_input}.")

        residue_keys_for_tasks = [
            (protein_id_input, row['chain'], row['residue'], row['resname_letter'], row['bfactor'])
            for _, row in residue_info_df.iterrows()
        ]
        print(f"Step 3: Identified {len(residue_keys_for_tasks)} standard residues. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 3: {e}"); traceback.print_exc(); return None, None, time.perf_counter() - overall_start_time

    # --- 4. Prepare Shared Memory (Optimized) ---
    print("Step 4a: Preparing shared memory arrays...")
    current_step_start_time = time.perf_counter()
    
    # Convert coordinates to contiguous array
    all_coords_np_orig = np.ascontiguousarray(atom_df[['x', 'y', 'z']].to_numpy(dtype=np.float32))
    all_elements_np_orig = np.ascontiguousarray(atom_df['element'].to_numpy(dtype='<U4'))  # Using safer fixed-width strings
    
    # Create shared memory for coordinates and elements
    shm_coords = shared_memory.SharedMemory(create=True, size=all_coords_np_orig.nbytes)
    shm_elements = shared_memory.SharedMemory(create=True, size=all_elements_np_orig.nbytes)
    
    # Create numpy arrays that use the shared memory buffers
    coords_shape = all_coords_np_orig.shape
    coords_dtype = all_coords_np_orig.dtype
    elements_shape = all_elements_np_orig.shape
    elements_dtype = all_elements_np_orig.dtype
    
    # Copy data to shared memory arrays
    shared_coords_np = np.ndarray(coords_shape, dtype=coords_dtype, buffer=shm_coords.buf)
    shared_elements_np = np.ndarray(elements_shape, dtype=elements_dtype, buffer=shm_elements.buf)
    np.copyto(shared_coords_np, all_coords_np_orig)
    np.copyto(shared_elements_np, all_elements_np_orig)
    
    # Prepare chain-specific DataFrames needed by _process_residue_env_worker
    chain_atom_dfs_global = {chain: group for chain, group in atom_df.groupby('chain')}
    
    print(f"Step 4a: Shared memory prepared. Time: {time.perf_counter() - current_step_start_time:.4f}s")

    # --- 4b. Process Residue Environments in Parallel ---
    print(f"Step 4b: Processing {len(residue_keys_for_tasks)} residue environments with {num_workers} workers...")
    current_step_start_time = time.perf_counter()
    
    # Prepare worker arguments - shared memory info
    worker_args_list = []
    for residue_key in residue_keys_for_tasks:
        worker_args = (
            residue_key,
            shm_coords.name, coords_shape, coords_dtype,
            shm_elements.name, elements_shape, elements_dtype,
            chain_atom_dfs_global, kdtree, env_radius
        )
        worker_args_list.append(worker_args)
    
    # Process environments in parallel
    with multiprocessing.Pool(num_workers, initializer=worker_init_fn) as pool:
        # Use imap instead of map to get results as they complete
        # This avoids accumulating all results in memory at once
        worker_results_iter = pool.imap(global_worker_task_executor, worker_args_list)
        
        # Collect results - gather valid ones
        cpu_pool_results = []
        for result in worker_results_iter:
            if result is not None and result[0] is not None:
                cpu_pool_results.append(result)
    
    print(f"Step 4b: Processed residue environments. Valid results: {len(cpu_pool_results)}. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    
    # Clean up shared memory
    try:
        shm_coords.close()
        shm_coords.unlink()
        shm_elements.close()
        shm_elements.unlink()
    except Exception as e_shm:
        print(f"Warning: Error cleaning up shared memory: {e_shm}")
    
    # --- 4.5 Create GPU Graphs (Main Process) ---
    if not cpu_pool_results:
        print("Error: No environments identified by workers.")
        return None, None, time.perf_counter() - overall_start_time

    print(f"Step 4.5: Creating {len(cpu_pool_results)} PyG graphs on GPU ({device_main_process})...")
    current_step_start_time = time.perf_counter()
    gpu_graphs_list = []
    final_metadata_list = []

    for result_item in cpu_pool_results:
        # Expected item: (metadata, env_data)
        metadata, env_data = result_item
        if metadata is None or env_data is None:
            continue
            
        env_coords_np, env_elements_np = env_data
            
        gpu_graph = create_pyg_graph_on_gpu(env_coords_np, env_elements_np,
                                          device_main_process, edge_cutoff, num_rbf)
        if gpu_graph:
            gpu_graphs_list.append(gpu_graph)
            final_metadata_list.append(metadata)

    if not gpu_graphs_list:
        print("Error: Failed to create any graphs on GPU from worker results.")
        return None, (final_metadata_list if final_metadata_list else None), time.perf_counter() - overall_start_time
    
    print(f"Step 4.5: Created {len(gpu_graphs_list)} graphs on GPU. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    metadata_list = final_metadata_list

    # --- 5. Batch Graphs (Similar to Ray version) ---
    graph_batch_gpu = None
    current_step_start_time = time.perf_counter()
    try:
        graph_batch_gpu = Batch.from_data_list(gpu_graphs_list)
        graph_batch_gpu = graph_batch_gpu.to(device_main_process)
        print(f"Step 5: Batching complete. Batch is on device: {graph_batch_gpu.x.device}. Time: {time.perf_counter() - current_step_start_time:.4f}s")
    except Exception as e:
        print(f"Error in Step 5: {e}"); traceback.print_exc()
        return None, metadata_list, time.perf_counter() - overall_start_time
    finally:
        if gpu_graphs_list: del gpu_graphs_list

    # --- 6. Run Model Inference (Same as Ray version) ---
    print(f"Step 6: Running inference on device {device_main_process}...")
    current_step_start_time = time.perf_counter()
    embeddings_np = None
    try:
        model_setup_time = time.perf_counter()
        print(f"  Moving model to {device_main_process} (if not already there)...")
        model_main_process.to(device_main_process)
        print(f"  Model on device. Time: {time.perf_counter() - model_setup_time:.4f}s")

        if COMPILE_MODEL and hasattr(torch, 'compile') and not isinstance(model_main_process, torch.jit.ScriptModule) and not model_main_process.__class__.__name__.endswith("CompiledModule"):
            compile_start_time = time.perf_counter()
            print("  Compiling model on target device (this may take a moment)...")
            try:
                model_main_process = torch.compile(model_main_process, mode="reduce-overhead")
                print(f"  Model compiled successfully on target device. Time: {time.perf_counter() - compile_start_time:.4f}s")
            except Exception as e_compile:
                print(f"  Warning: Model compilation on {device_main_process} failed: {e_compile}")
        
        actual_inference_start_time = time.perf_counter()
        model_main_process.eval()
        with torch.no_grad():
            with torch.autocast(device_type=str(device_main_process.type), dtype=torch.float16, enabled=(str(device_main_process.type) == 'cuda')):
                embs, _ = model_main_process.online_encoder(graph_batch_gpu, return_projection=False)
                embeddings_np = embs.float().cpu().numpy()
        
        print(f"  Actual inference. Time: {time.perf_counter() - actual_inference_start_time:.4f}s")
        print(f"Step 6: Inference finished. Output shape: {embeddings_np.shape}. Total Step 6 Time: {time.perf_counter() - current_step_start_time:.4f}s")
        if embeddings_np.shape[0] != len(metadata_list):
            raise ValueError(f"Embedding count mismatch: Expected {len(metadata_list)}, Got {embeddings_np.shape[0]}.")
    except RuntimeError as e_rt:
        if "CUDA out of memory" in str(e_rt): print("CUDA OOM during inference!")
        else: print(f"Runtime error during inference: {e_rt}")
        traceback.print_exc()
        return None, metadata_list, time.perf_counter() - overall_start_time
    except Exception as e_inf:
        print(f"Error in Step 6 (Inference): {e_inf}"); traceback.print_exc()
        return None, metadata_list, time.perf_counter() - overall_start_time
    finally:
        if graph_batch_gpu is not None: del graph_batch_gpu
        if 'embs' in locals(): del embs
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 7. Return Results ---
    total_time_taken = time.perf_counter() - overall_start_time
    print("-" * 30)
    print(f"OPTIMIZED pipeline finished successfully for {protein_id_input}.")
    print(f"Total processing time (perf_counter): {total_time_taken:.4f} seconds.")
    print("-" * 30)
    return embeddings_np, metadata_list, total_time_taken

def run_pipeline():
    # Set up device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available(): print(f"Found {torch.cuda.device_count()} CUDA devices.")

    print("Loading model for main process...")
    # Model is loaded once here and passed to generate_single_pdb_embedding functions
    # It stays on the CPU initially, or moved to device later as needed
    model_main = initialize_model(CHECKPOINT_PATH, device=cpu_device) # Keep on CPU first
    model_main.eval()
    print("Model loaded on CPU for main process.")

    print(f"Loading PDB file for testing: {PDB_FILE_PATH}...")
    try:
        atom_df_for_testing = fo.bp_to_df(fo.read_any(PDB_FILE_PATH))
        protein_id_for_testing = os.path.splitext(os.path.basename(PDB_FILE_PATH))[0]
        if atom_df_for_testing is None or atom_df_for_testing.empty:
            raise ValueError("Loaded DataFrame for testing is empty.")
        print(f"Successfully loaded PDB. Atoms: {len(atom_df_for_testing)}")
    except Exception as e:
        print(f"Failed to load PDB for testing: {e}"); traceback.print_exc(); return

    if atom_df_for_testing is not None:
        profiler = cProfile.Profile()
        profiler.enable()

        # Benchmark all implementations for comparison
        print("\n=== PERFORMANCE BENCHMARKS ===")
        
        # 1. Run optimized (threads) implementation - should be fastest for most cases
        print("\n[1] RUNNING OPTIMIZED (THREAD-BASED) IMPLEMENTATION:")
        embeddings_opt, metadata_opt, duration_opt = generate_single_pdb_embedding_optimized(
            atom_df_for_testing,
            protein_id_for_testing,
            model_main,
            device,
            include_hets=INCLUDE_HETS,
            env_radius=ENV_RADIUS,
            edge_cutoff=EDGE_CUTOFF,
            num_rbf=NUM_RBF,
            num_workers=NUM_WORKERS,
            use_ray=False  # Force thread-based implementation
        )
        
        print(f"\nTHREAD-BASED PERFORMANCE: {duration_opt:.3f} seconds")
        if embeddings_opt is not None:
            print(f"Thread-based generated {len(metadata_opt)} embeddings of shape {embeddings_opt.shape}")
        
        # 2. Run Ray implementation for comparison
        print("\n[2] RUNNING RAY IMPLEMENTATION FOR COMPARISON:")
        embeddings_ray, metadata_ray, duration_ray = generate_single_pdb_embedding_optimized(
            atom_df_for_testing,
            protein_id_for_testing,
            model_main,
            device,
            include_hets=INCLUDE_HETS,
            env_radius=ENV_RADIUS,
            edge_cutoff=EDGE_CUTOFF,
            num_rbf=NUM_RBF,
            num_workers=NUM_WORKERS,
            use_ray=True  # Force Ray implementation
        )
        
        print(f"\nRAY-BASED PERFORMANCE: {duration_ray:.3f} seconds")
        if embeddings_ray is not None:
            print(f"Ray-based generated {len(metadata_ray)} embeddings of shape {embeddings_ray.shape}")
        
        # Compare results
        if embeddings_opt is not None and embeddings_ray is not None:
            # This is not necessary but helps validate implementations match
            try:
                embedding_similarity = np.mean(np.abs(embeddings_opt - embeddings_ray))
                print(f"\nEmbedding similarity (MAE): {embedding_similarity:.6f}")
                if embedding_similarity < 1e-5:
                    print("Implementations produce numerically equivalent results (as expected)")
                else:
                    print("WARNING: Implementations produce slightly different results")
            except:
                print("Could not compare embeddings (different shapes or one is None)")
        
        print("\n=== BENCHMARK SUMMARY ===")
        print(f"Thread-based: {duration_opt:.3f}s, Ray-based: {duration_ray:.3f}s")
        print(f"Speedup: {duration_ray/duration_opt:.2f}x faster with optimized implementation")
        
        # Use the optimized implementation results
        embeddings, metadata, duration = embeddings_opt, metadata_opt, duration_opt

        profiler.disable()
        print("Programmatic profiling finished.")

        if embeddings is not None and metadata is not None:
            print("\n--- Pipeline Results ---")
            print(f"Successfully generated embeddings for {len(metadata)} residues.")
            print(f"Final Embeddings Shape: {embeddings.shape}")
            print(f"Total Duration: {duration:.3f} seconds")

            print("\nMetadata example (first 5 residues):")
            for i in range(min(5, len(metadata))):
                meta_copy = metadata[i].copy()
                meta_copy['confidence'] = f"{meta_copy.get('confidence', 0.0):.2f}"
                print(f"  {i+1}: {meta_copy}")

            if len(embeddings) > 0:
                 print("\nEmbedding example (first residue stats):")
                 print(f"  Min: {np.min(embeddings[0]):.3f}, Max: {np.max(embeddings[0]):.3f}, Mean: {np.mean(embeddings[0]):.3f}")
        else:
            print("\n--- Pipeline Failed ---")
            print("Embedding generation was unsuccessful. Check logs above for errors.")

    if 'profiler' in locals():
        print("\n--- Profiler Stats (Top 20 by Total Time) ---")
        stats_tottime = pstats.Stats(profiler).sort_stats('tottime')
        stats_tottime.print_stats(20)

        print("\n--- Profiler Stats (Top 20 by Cumulative Time) ---")
        stats_cumtime = pstats.Stats(profiler).sort_stats('cumtime')
        stats_cumtime.print_stats(20)
        
        profiler_output_file = "programmatic_profile_optimized.prof"
        profiler.dump_stats(profiler_output_file)
        print(f"\nFull profiler data saved to {profiler_output_file}")

if __name__ == '__main__':
    # Define cpu_device here for global access
    cpu_device = torch.device('cpu')
    
    # CPU count validation: safety check
    available_cpus = os.cpu_count()
    if NUM_WORKERS > available_cpus:
        print(f"Warning: NUM_WORKERS ({NUM_WORKERS}) exceeds available CPU count ({available_cpus})")
        print(f"Reducing worker count to {available_cpus}")
        NUM_WORKERS = available_cpus
    
    # Run the benchmark pipeline with both implementations
    run_pipeline()
