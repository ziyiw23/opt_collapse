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
import multiprocessing
from functools import partial
import traceback # For printing detailed errors

# --- Imports from project ---
# Assuming collapse and embedding_utils are importable from the notebook's environment
# Adjust paths if necessary
try:
    from collapse import atom_info, initialize_model # initialize_model loads model + BYOL wrapper
    from atom3d.filters.filters import first_model_filter # Preprocessing
    import atom3d.util.formats as fo # For reading PDB

    # Import necessary functions directly from embedding_utils
    from embedding_utils import (
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

# %% [markdown]
# ## Configuration

# %%
# --- Parameters ---
# <<< USER: SET YOUR PDB FILE HERE >>>
PDB_FILE_PATH = "/scratch/groups/rbaltman/ziyiw23/clps_pdbs/MGYP000002588102.pdb"
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
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        print(f"Set multiprocessing start method to 'spawn'.")
except Exception as e:
    # Ignore if it's already 'spawn' or cannot be changed (e.g., in certain envs)
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != 'spawn':
        print(f"Warning: Could not force multiprocessing start method to 'spawn' ('{e}'). Using default: '{current_method}'.")
    else:
        print(f"Multiprocessing start method already '{current_method}'.")


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

def create_pyg_graph(df_env, edge_cutoff=EDGE_CUTOFF, num_rbf=NUM_RBF, device='cpu'):
    """
    Creates a PyG Data object from a DataFrame representing a residue's environment.
    Simplified and adapted from embedding_utils.BaseTransform.
    Operates on CPU, graphs will be moved to GPU in batch later.
    """
    graph_device = torch.device('cpu') # Force graph creation on CPU

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

def _process_residue_env_worker(residue_key, full_atom_df_global, chain_atom_dfs_global, kdtree_global, env_radius, edge_cutoff, num_rbf):
    """
    Worker function executed by multiprocessing pool to generate graph for one residue.
    Accesses global-like variables passed via partial or closure.

    Args:
        residue_key (tuple): (protein_id, chain_id, resnum, resname_letter, bfactor)
        full_atom_df_global (pd.DataFrame): The complete, preprocessed atomic dataframe.
        chain_atom_dfs_global (dict): Dict mapping chain_id -> DataFrame for that chain.
        kdtree_global (scipy.spatial.cKDTree): Pre-built KDTree on full_atom_df_global coords.
        env_radius (float): Radius for KDTree neighbor search.
        edge_cutoff (float): Cutoff for graph edges within the environment.
        num_rbf (int): Number of radial basis functions for edge features.

    Returns:
        tuple: (metadata_dict, pyg_graph) or (None, None) if processing fails.
    """
    protein_id, chain_id, resnum, resname_letter, bfactor = residue_key
    resid_tuple = (resname_letter, resnum) # Required format for sample_functional_center

    # Access the specific chain's DataFrame
    chain_atoms_df = chain_atom_dfs_global.get(chain_id)
    if chain_atoms_df is None:
        # This shouldn't happen if keys are derived correctly, but check defensively.
        # print(f"Debug: Chain DF not found for {chain_id} in worker.")
        return None, None

    # --- Get Atoms for the Target Residue (for center calculation) ---
    # Filter the chain-specific DataFrame more efficiently
    res_df = chain_atoms_df[chain_atoms_df['residue'] == resnum]
    if res_df.empty:
        # print(f"Debug: Residue {resnum} not found in chain_df for {chain_id}.")
        return None, None

    # --- Calculate Residue Center ---
    center = sample_functional_center(res_df, resid_tuple, train_mode=False)
    if center is None:
        # Failed to find functional atoms or CA atom.
        # print(f"Debug: Could not calculate center for {resid_tuple} in {protein_id}.")
        return None, None

    # --- Find Neighboring Atoms using KDTree ---
    try:
        # Query the pre-built KDTree (kdtree_global) using the calculated center.
        # It returns indices corresponding to the rows in full_atom_df_global.
        pt_indices = kdtree_global.query_ball_point(center, r=env_radius)

        # If no neighbors are found within the radius:
        if not pt_indices:
             # print(f"Debug: No neighbors found within radius {env_radius} for {resid_tuple} in {protein_id}.")
             return None, None

        # Select the rows corresponding to the neighbor indices from the full DataFrame.
        # Use .iloc as pt_indices are integer indices from the KDTree query.
        # Important: kdtree_global must have been built on full_atom_df_global *after* reset_index(drop=True).
        df_env = full_atom_df_global.iloc[pt_indices].copy() # Use copy to avoid modifying global df slice

        # If the resulting environment DataFrame is empty (shouldn't happen if pt_indices is not empty):
        if df_env.empty:
             return None, None

    except Exception as e:
        print(f"Error during KDTree query/neighbor selection for {resid_tuple} in {protein_id}: {e}\n{traceback.format_exc()}")
        return None, None

    # --- Create PyG Graph from Environment ---
    # Graph is created on CPU within the worker
    graph = create_pyg_graph(df_env, edge_cutoff=edge_cutoff, num_rbf=num_rbf, device='cpu')

    # If graph creation failed:
    if graph is None:
        # print(f"Debug: Graph creation failed for residue {resid_tuple}")
        return None, None

    # --- Prepare Metadata ---
    # Create a dictionary holding information about this residue environment
    metadata = {
        'protein_id': protein_id,
        'chain': chain_id,
        'resid': f"{resname_letter}{resnum}", # Standard residue ID (e.g., A123)
        'resname_letter': resname_letter, # Single letter AA code
        'resnum': resnum, # Residue number
        'confidence': bfactor # B-factor, often pLDDT from AlphaFold
    }

    # Return the metadata and the generated graph
    return metadata, graph

# %% [markdown]
# ## Main Embedding Generation Function

# %%
def generate_single_pdb_embedding(atom_df_input: pd.DataFrame, protein_id_input: str, # MODIFIED: New input arguments
                                  model, device,
                                  include_hets=INCLUDE_HETS,
                                  env_radius=ENV_RADIUS,
                                  edge_cutoff=EDGE_CUTOFF,
                                  num_rbf=NUM_RBF,
                                  num_workers=NUM_WORKERS):
    """
    Orchestrates the process of generating COLLAPSE embeddings for a single PDB's atomic data.

    Steps:
    1. Preprocess the input atomic DataFrame.
    2. Build a KDTree for efficient spatial queries.
    3. Identify standard amino acid residues to process.
    4. Use multiprocessing to generate PyG graphs for each residue's environment in parallel.
    5. Batch the generated graphs.
    6. Run inference on the batch using the provided COLLAPSE model.
    7. Return embeddings, metadata, and processing time.

    Args:
        atom_df_input (pd.DataFrame): DataFrame containing the atomic data for the structure.
                                      Expected to have columns like 'x', 'y', 'z', 'element', 'resname', etc.
        protein_id_input (str): An identifier for the protein/structure.
        model (torch.nn.Module): The pre-loaded (and potentially compiled) COLLAPSE model.
        device (torch.device): The device (CPU or CUDA) for model inference.
        include_hets (bool): Flag to include heteroatoms during preprocessing.
        env_radius (float): Radius (Angstroms) to define the residue environment.
        edge_cutoff (float): Distance cutoff (Angstroms) for graph edges.
        num_rbf (int): Number of radial basis functions for edge features.
        num_workers (int): Number of CPU cores to use for parallel graph generation.

    Returns:
        tuple: (embeddings_np, metadata_list, processing_time_seconds)
               Returns (None, None, time) if any critical step fails.
               `embeddings_np` is a NumPy array (num_residues, embedding_dim).
               `metadata_list` is a list of dictionaries, one per residue.
               `processing_time_seconds` is the total time taken.
    """
    pipeline_start_time = time.time()
    print("-" * 30)
    print(f"Starting embedding generation for: {protein_id_input}") # MODIFIED: Use protein_id_input
    print(f"Parameters: EnvRadius={env_radius}, EdgeCutoff={edge_cutoff}, IncludeHets={include_hets}, Workers={num_workers}")
    print("-" * 30)

    # --- 1. Load and Preprocess PDB ---
    # MODIFIED: Use protein_id_input directly and work with atom_df_input
    protein_id = protein_id_input
    atom_df = None # Initialize df
    try:
        # Use a copy of the input DataFrame to avoid modifying the original
        atom_df = atom_df_input.copy()
        if atom_df is None or atom_df.empty:
             raise ValueError(f"Input atom_df_input is None or empty for {protein_id}")

        # Apply standard preprocessing filters
        atom_df = first_model_filter(atom_df) # Keep only first model
        # del atoms_df_raw # No raw_df to delete here
        atom_df = atom_df[~atom_df.hetero.str.contains('W', na=False)] # Remove water molecules
        atom_df = atom_df[atom_df['element'] != 'H'] # Remove hydrogen atoms

        # Optionally filter out heteroatoms/non-standard residues
        if not include_hets:
            if hasattr(atom_info, 'aa') and isinstance(atom_info.aa, (list, set)):
                if 'resname' in atom_df.columns:
                    atom_df = atom_df[atom_df.resname.isin(atom_info.aa)]
                else:
                    print(f"Warning: 'resname' column missing, cannot filter hets for {protein_id}.")
            else:
                print("Warning: atom_info.aa not found, cannot filter hets.")

        # Reset index *after* all filtering - important for iloc later
        atom_df = atom_df.reset_index(drop=True)
        if atom_df.empty:
            raise ValueError(f"PDB empty after filtering: {protein_id}")

        # Check for essential columns needed downstream
        essential_cols = {'resname', 'chain', 'residue', 'element', 'x', 'y', 'z', 'name', 'bfactor'}
        missing_cols = essential_cols - set(atom_df.columns)
        if missing_cols:
             raise ValueError(f"Missing essential columns after filtering in {protein_id}: {missing_cols}")

        atom_df['id'] = protein_id # Add protein ID column (optional, for clarity)
        print(f"Step 1: Loaded and preprocessed. Atoms: {len(atom_df)}")

    except Exception as e:
        print(f"Error in Step 1 (Load/Preprocess) for {protein_id}: {e}")
        print(traceback.format_exc())
        return None, None, time.time() - pipeline_start_time

    # --- 2. Build KDTree for Spatial Queries ---
    kdtree = None
    try:
        # Extract coordinates as a NumPy array for KDTree
        coords_np = atom_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        if coords_np.shape[0] == 0: raise ValueError("No coordinates available for KDTree.")

        # Build the KDTree using SciPy
        kdtree = scipy.spatial.cKDTree(coords_np, compact_nodes=True, copy_data=False) # Optimizations
        del coords_np # Free memory
        print(f"Step 2: Built KDTree for {kdtree.n} atoms.")
    except Exception as e:
        print(f"Error in Step 2 (Build KDTree) for {protein_id}: {e}")
        print(traceback.format_exc())
        return None, None, time.time() - pipeline_start_time

    # --- 3. Identify Residues and Prepare for Parallel Processing ---
    residue_keys = []
    chain_atom_dfs = {}
    try:
        # Map full residue names (e.g., 'ALA') to single letters ('A')
        if atom_info.aa_to_letter_dict:
            # Use precomputed dictionary if available (faster)
             # Ensure all resnames are present in the dict, handle missing ones if necessary
            unknown_res = set(atom_df['resname'].unique()) - set(atom_info.aa_to_letter_dict.keys())
            if unknown_res:
                # If only filtering standard AAs, this might be ok if they are non-AA hets
                # But if 'include_hets' was True, this mapping might fail.
                # Let's map known ones and potentially handle/warn about others later
                atom_df['resname_letter'] = atom_df['resname'].map(atom_info.aa_to_letter_dict)
                # print(f"Debug: Unknown resnames encountered during mapping: {unknown_res}")
            else:
                atom_df['resname_letter'] = atom_df['resname'].map(atom_info.aa_to_letter_dict)
        else:
            # Fallback to using the function call (slower)
            atom_df['resname_letter'] = atom_df['resname'].apply(atom_info.aa_to_letter)


        # Get unique standard residues: (chain, residue_number, single_letter_code, bfactor)
        # Keep only the first b-factor encountered for each residue (can be mean/median if needed)
        # Ensure 'residue' column is numeric for proper comparison/grouping if needed
        try:
            atom_df['residue'] = pd.to_numeric(atom_df['residue'])
        except ValueError:
             print("Warning: 'residue' column contains non-numeric values.")
             # Handle appropriately - maybe filter these rows out earlier?

        residue_info = atom_df[['chain', 'residue', 'resname_letter', 'bfactor']].drop_duplicates(subset=['chain', 'residue'])

        # Filter to keep only standard amino acids (using atom_info)
        standard_letters = set(atom_info.aa_abbr) - {'X'} if hasattr(atom_info, 'aa_abbr') else set('ACDEFGHIKLMNPQRSTVWY')
        # Also filter out rows where resname_letter might be NaN due to mapping issues
        residue_info = residue_info[residue_info['resname_letter'].isin(standard_letters) & residue_info['resname_letter'].notna()]


        if residue_info.empty:
            raise ValueError(f"No standard amino acid residues found after filtering in {protein_id}.")

        # Create the list of keys for the multiprocessing map function
        residue_keys = [
            (protein_id, row['chain'], row['residue'], row['resname_letter'], row['bfactor'])
            for _, row in residue_info.iterrows()
        ]
        del residue_info # Free memory

        # Pre-split the main DataFrame by chain for faster access in worker processes
        chain_atom_dfs = {chain_id: group_df for chain_id, group_df in atom_df.groupby('chain')}

        print(f"Step 3: Identified {len(residue_keys)} standard residues to process.")

    except Exception as e:
        print(f"Error in Step 3 (Identify Residues) for {protein_id}: {e}")
        print(traceback.format_exc())
        # Clean up potentially large objects even if this step fails
        if 'atom_df' in locals() and atom_df is not None: del atom_df
        if kdtree is not None: del kdtree
        return None, None, time.time() - pipeline_start_time

    # --- 4. Generate Graphs in Parallel using Multiprocessing ---
    graphs = []
    metadata_list = []
    print(f"Step 4: Generating residue graphs using {num_workers} workers...")
    graph_gen_start_time = time.time()

    # Use functools.partial to create a worker function with fixed arguments
    # This avoids pickling large objects like DataFrames repeatedly for each task.
    # The Pool passes only the unique `residue_key` to each worker call.
    worker_func_partial = partial(
        _process_residue_env_worker,
        full_atom_df_global=atom_df,    # Pass the full DataFrame
        chain_atom_dfs_global=chain_atom_dfs, # Pass the chain DataFrame lookup
        kdtree_global=kdtree,         # Pass the KDTree
        env_radius=env_radius,
        edge_cutoff=edge_cutoff,
        num_rbf=num_rbf
    )

    results = []
    try:
        # Create a multiprocessing pool and map the worker function over residue keys
        if num_workers > 0 and len(residue_keys) > 1: # No point using pool for 1 residue or 0 workers
             # Consider chunking if number of residues is very large? pool.map handles distribution.
             # chunksize = max(1, len(residue_keys) // (num_workers * 4)) # Example chunking
             with multiprocessing.Pool(processes=num_workers) as pool:
                 # results = pool.map(worker_func_partial, residue_keys, chunksize=chunksize)
                 results = pool.map(worker_func_partial, residue_keys)
        else: # Run sequentially for debugging or if num_workers is 0 or only 1 task
             if num_workers <= 0: print("Running graph generation sequentially (num_workers <= 0).")
             elif len(residue_keys) <=1 : print(f"Running graph generation sequentially ({len(residue_keys)} residue).")
             results = [worker_func_partial(key) for key in residue_keys]


        # Process the results, filtering out any failures (None, None pairs)
        valid_results = [res for res in results if res is not None and res[0] is not None and res[1] is not None]

        if not valid_results:
             raise ValueError(f"No valid graphs were generated for {protein_id} after parallel processing.")

        # Unzip the valid results into separate lists for metadata and graphs
        metadata_list, graphs = zip(*valid_results)
        metadata_list = list(metadata_list) # Convert tuple to list
        graphs = list(graphs)          # Convert tuple to list

        graph_gen_duration = time.time() - graph_gen_start_time
        print(f"Step 4: Generated {len(graphs)} graphs in {graph_gen_duration:.2f} seconds.")

    except Exception as e:
        print(f"Error in Step 4 (Parallel Graph Generation) for {protein_id}: {e}")
        print(traceback.format_exc())
        return None, None, time.time() - pipeline_start_time
    finally:
        # Clean up large objects passed to partial function
        del worker_func_partial
        if kdtree is not None: del kdtree
        if chain_atom_dfs: del chain_atom_dfs
        if atom_df is not None: del atom_df # Release DataFrame memory


    # --- 5. Batch Graphs and Move to Inference Device ---
    if not graphs:
        print("Error: No graphs available for batching.")
        return None, None, time.time() - pipeline_start_time

    print(f"Step 5: Batching {len(graphs)} graphs and moving to device {device}...")
    graph_batch = None
    try:
        # Use PyG's Batch class to collate the list of Data objects into a single graph batch
        graph_batch = Batch.from_data_list(graphs).to(device)
        print(f"Step 5: Batching complete.")
    except Exception as e:
        print(f"Error in Step 5 (Batching Graphs): {e}")
        print(traceback.format_exc())
        # Return metadata if available, but embeddings failed
        return None, metadata_list, time.time() - pipeline_start_time
    finally:
         # Graphs list might contain many Data objects, clear it after batching
         if graphs: del graphs

    # --- 6. Run Model Inference ---
    print(f"Step 6: Running inference on device {device}...")
    inference_start_time = time.time()
    embeddings_np = None
    try:
        # --- MODIFICATION: Move model to GPU and compile just before inference ---
        print(f"Moving model to {device}...")
        model.to(device)
        print("Model moved.")

        # Compile the model *after* moving to the target device
        if COMPILE_MODEL and hasattr(torch, 'compile'):
            print("Compiling model on target device (this may take a moment)...")
            try:
                # Use a mode suitable for inference after potential graph changes
                model = torch.compile(model, mode="reduce-overhead") # or "max-autotune"
                print("Model compiled successfully on target device.")
            except Exception as e:
                print(f"Warning: Model compilation on {device} failed: {e}")
        # --- END MODIFICATION ---


        # Ensure model is in eval mode and no gradients are computed
        model.eval()
        with torch.no_grad():
            # Use autocast for potential speedup with mixed precision (especially on CUDA)
            with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                # --- MODIFICATION: Call online_network directly for inference --- 
                # The BYOL wrapper's main forward method might be for training.
                # Access the online network attribute directly.
                # embs = model.online_network(graph_batch)
                # --- MODIFIED: Use online_encoder as seen in other parts of the codebase ---
                embs, _ = model.online_encoder(graph_batch, return_projection=False)
                # --- END MODIFICATION ---

                # Check output dimension if needed (e.g., embs.shape[-1])
                # Expected shape: (num_residues, projection_size) e.g., (N, 512)

                # Convert embeddings to float32 and move to CPU for NumPy conversion
                embeddings_np = embs.float().cpu().numpy()

        inference_duration = time.time() - inference_start_time
        print(f"Step 6: Inference finished in {inference_duration:.2f} seconds. Output shape: {embeddings_np.shape}")

        # --- Verification ---
        # Check if the number of output embeddings matches the number of input residues processed
        if embeddings_np.shape[0] != len(metadata_list):
            raise ValueError(f"Embedding count mismatch: Expected {len(metadata_list)}, Got {embeddings_np.shape[0]}.")

    except RuntimeError as e:
         if "CUDA out of memory" in str(e):
             print("CUDA Out of Memory Error during inference!")
             print("Input batch size (residues):", graph_batch.num_graphs)
             print("Try reducing model size/precision or using a GPU with more memory.")
         else:
             print(f"Runtime error during inference: {e}")
         print(traceback.format_exc())
         return None, metadata_list, time.time() - pipeline_start_time
    except Exception as e:
        print(f"Error in Step 6 (Inference): {e}")
        print(traceback.format_exc())
        return None, metadata_list, time.time() - pipeline_start_time
    finally:
        # Clean up GPU memory explicitly
        if graph_batch is not None: del graph_batch
        if 'embs' in locals(): del embs
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 7. Return Results ---
    total_time = time.time() - pipeline_start_time
    print("-" * 30)
    print(f"Pipeline finished successfully for {protein_id}.")
    print(f"Total processing time: {total_time:.3f} seconds.")
    print("-" * 30)
    return embeddings_np, metadata_list, total_time

# --- MODIFICATION: Add __name__ == '__main__' guard ---
if __name__ == '__main__':
    # Set multiprocessing start method (important for CUDA compatibility with multiprocessing)
    # 'spawn' is generally recommended when using CUDA.
    # Moved here to ensure it runs only in the main process before Pool creation
    try:
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            print(f"Set multiprocessing start method to 'spawn'.")
    except Exception as e:
        # Ignore if it's already 'spawn' or cannot be changed (e.g., in certain envs)
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            print(f"Warning: Could not force multiprocessing start method to 'spawn' ('{e}'). Using default: '{current_method}'.")
        # else:
            # print(f"Multiprocessing start method already '{current_method}'.") # Optional: less verbose

    # --- Moved device setup prints here ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices.")
    # --- End moved device setup prints ---

    if not os.path.exists(PDB_FILE_PATH):
        print(f"Error: PDB file not found at {PDB_FILE_PATH}")
        print("Please update the PDB_FILE_PATH variable in the 'Configuration' cell.")
    else:
        # --- MODIFICATION: Load PDB into DataFrame before calling the function ---
        print(f"Loading PDB file for testing: {PDB_FILE_PATH}...")
        try:
            atom_df_for_testing = fo.bp_to_df(fo.read_any(PDB_FILE_PATH))
            protein_id_for_testing = os.path.splitext(os.path.basename(PDB_FILE_PATH))[0]

            if atom_df_for_testing is None or atom_df_for_testing.empty:
                raise ValueError("Loaded DataFrame for testing is empty.")
            print(f"Successfully loaded PDB into DataFrame. Atoms: {len(atom_df_for_testing)}")

        except Exception as e:
            print(f"Failed to load PDB for testing: {e}")
            print(traceback.format_exc())
            atom_df_for_testing = None # Ensure it's None if loading fails

        if atom_df_for_testing is not None:
            # --- END MODIFICATION ---

            # --- Run the main pipeline function ---
            # Pass the target device, the model will be moved inside the function
            embeddings, metadata, duration = generate_single_pdb_embedding(
                atom_df_for_testing, # MODIFIED: Pass DataFrame
                protein_id_for_testing, # MODIFIED: Pass protein_id
                model, # Model starts on CPU
                device, # Target device for inference
                include_hets=INCLUDE_HETS,
                env_radius=ENV_RADIUS,
                edge_cutoff=EDGE_CUTOFF,
                num_rbf=NUM_RBF,
                num_workers=NUM_WORKERS
            )

            # --- Output Results ---
            if embeddings is not None and metadata is not None:
                print("\n--- Pipeline Results ---")
                print(f"Successfully generated embeddings for {len(metadata)} residues.")
                print(f"Final Embeddings Shape: {embeddings.shape}")
                print(f"Total Duration: {duration:.3f} seconds")

                # Example: Display metadata for the first few residues
                print("\nMetadata example (first 5 residues):")
                for i in range(min(5, len(metadata))):
                    # Format confidence (bfactor) to 2 decimal places
                    meta_copy = metadata[i].copy()
                    meta_copy['confidence'] = f"{meta_copy.get('confidence', 0.0):.2f}"
                    print(f"  {i+1}: {meta_copy}")

                # Example: Display embedding statistics for the first residue
                if len(embeddings) > 0:
                     print("\nEmbedding example (first residue stats):")
                     print(f"  Min: {np.min(embeddings[0]):.3f}, Max: {np.max(embeddings[0]):.3f}, Mean: {np.mean(embeddings[0]):.3f}")

            else:
                print("\n--- Pipeline Failed ---")
                print("Embedding generation was unsuccessful. Check logs above for errors.")
