# embedding_utils.py

import numpy as np
import torch
import pandas as pd
# Use scipy.spatial directly to call cKDTree
import scipy.spatial
# Import torch_cluster needed for BaseTransform
import torch_cluster
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
# Import filter needed for preprocessing in the parallel framework
from atom3d.filters.filters import first_model_filter

# Assuming atom_info is importable and has the required structure
# Ensure atom_info has necessary attributes like aa_to_letter, abbr_key_atom_dict, aa
from collapse import atom_info

# =================== Constants ===================

ELEMENT_MAPPING = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'CL': 5,
    'P': 6, 'Se': 7, 'SE': 7, 'Fe': 8, 'FE': 8, 'Zn': 9, 'ZN': 9,
    'Ca': 10, 'CA': 10, 'Mg': 11, 'MG': 11,
}
DEFAULT_ELEMENT = 12

# =================== Helper Functions ===================

def _element_mapping(x):
    """Maps element symbol string to integer."""
    # Ensure input is string and handle potential NaN/None
    if not isinstance(x, str):
        return DEFAULT_ELEMENT
    return ELEMENT_MAPPING.get(x.strip().upper(), DEFAULT_ELEMENT)

def _normalize(tensor, dim=-1):
    """Normalizes a torch.Tensor along a dimension without NaNs."""
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """Radial Basis Function embedding."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    """Calculates scalar and vector edge features."""
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    distances = torch.norm(E_vectors, dim=-1)
    rbf = _rbf(distances, D_max=D_max, D_count=num_rbf, device=device)
    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_s, edge_v

# =================== Graph Construction ===================

class BaseTransform:
    """
    Creates graph from a DataFrame subset using torch_cluster.radius_graph.
    Matches the structure of the BaseTransform in collapse.data.
    Designed to be instantiated and used by residue processing functions.
    """
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device

    def __call__(self, df):
        """ Creates graph using torch_cluster.radius_graph """
        protein_id = df['id'].iloc[0] if 'id' in df and not df.empty else 'graph_gen'
        try:
            with torch.no_grad():
                # Map elements, ensuring 'element' column exists and handling potential errors
                if 'element' not in df.columns:
                    print(f"Warning: 'element' column missing in df for {protein_id}")
                    return None
                atoms = torch.as_tensor(list(map(_element_mapping, df['element'])),
                                        dtype=torch.long, device=self.device)

                # Get coordinates, ensuring columns exist
                if not {'x', 'y', 'z'}.issubset(df.columns):
                    print(f"Warning: coordinate columns missing in df for {protein_id}")
                    return None
                coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(dtype=np.float32),
                                         dtype=torch.float32, device=self.device)

                if coords.dim() == 1: coords = coords.unsqueeze(0)
                if coords.shape[0] == 0:
                    # print(f"DEBUG BaseTransform ({protein_id}): Exiting because coords shape[0] is 0.") # DEBUG
                    return None

                edge_index = torch_cluster.radius_graph(
                    coords,
                    r=self.edge_cutoff,
                    batch=None, # No batching within a single structure's transform
                    loop=False, # Match original pipeline behavior
                    flow='source_to_target'
                )

                if edge_index.shape[1] == 0:
                    # print(f"DEBUG BaseTransform ({protein_id}): Exiting because edge_index shape[1] is 0.") # DEBUG
                    return None

                edge_s, edge_v = _edge_features(coords, edge_index, D_max=self.edge_cutoff,
                                               num_rbf=self.num_rbf, device=self.device)

                data = Data(x=coords, atoms=atoms,
                           edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
                # Optionally add chain info if present, matching original BaseTransform
                if 'same_chain' in df.columns:
                    data.chain_ind = torch.as_tensor(df.same_chain.tolist(), dtype=torch.long, device=self.device)

                return data
        except Exception as e:
            print(f"Error during BaseTransform ({protein_id}): {e}")
            import traceback; traceback.print_exc()
            return None

# =================== Residue Processing ===================

def sample_functional_center(res_df, resid_tuple, train_mode=False):
    """Calculates the geometric center of functional atoms for a residue."""
    if res_df.empty: return None
    resname_letter, resnum = resid_tuple # Use both for clarity in potential errors

    # Determine functional atom names based on atom_info
    # Default to CA if specific atoms aren't defined or found
    default_func_atoms = ['CA']
    func_atoms = default_func_atoms # Start with default

    # Check if abbr_key_atom_dict exists and has the key
    if hasattr(atom_info, 'abbr_key_atom_dict') and resname_letter in atom_info.abbr_key_atom_dict:
        func_atoms_options = atom_info.abbr_key_atom_dict[resname_letter]
        if func_atoms_options: # Check if the list of options is not empty
             if not train_mode:
                 # Flatten list of lists and remove duplicates
                 specific_atoms = list(set(atom for sublist in func_atoms_options for atom in sublist))
                 if specific_atoms: # Use specific atoms if any were found
                      func_atoms = specific_atoms
             else: # train_mode=True - Sample one functional atom (Not currently used)
                 all_func_atoms = list(set(atom for sublist in func_atoms_options for atom in sublist))
                 if all_func_atoms:
                     func_atoms = [np.random.choice(all_func_atoms)]
                 # else func_atoms remains ['CA']

    # Calculate center
    try:
        # Ensure required columns exist
        if not {'x', 'y', 'z', 'name'}.issubset(res_df.columns):
             print(f"Warning: Missing columns in res_df for center calculation of {resname_letter}{resnum}")
             return None

        coords_all = res_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        names = res_df['name'].to_numpy()

        # Filter coords corresponding to functional atom names
        atom_mask = np.isin(names, func_atoms)
        func_coords = coords_all[atom_mask]

        # If no specified functional atoms found, explicitly try CA if it wasn't the default
        if func_coords.shape[0] == 0 and 'CA' not in func_atoms:
            ca_mask = (names == 'CA')
            if np.any(ca_mask):
                func_coords = coords_all[ca_mask]

        # If still no coordinates found (e.g., residue missing CA and functional atoms)
        if func_coords.shape[0] == 0:
             # print(f"Warning: Could not find functional atoms {func_atoms} or CA for {resname_letter}{resnum}")
             return None # Cannot determine center

        center = np.mean(func_coords, axis=0, dtype=np.float32)
        return center
    except Exception as e:
        print(f"Error calculating center for {resname_letter}{resnum}: {e}")
        return None

def extract_env_for_residue_cpu(atoms_df, chain_atoms_df, resid_tuple, env_radius, base_transform_cpu, return_debug_info=False):
    """
    Creates graph for one residue environment using a BaseTransform instance.
    Uses the FULL atoms_df for KDTree construction like the original pipeline.
    Intended to be called by worker processes or serial processing loops.

    Args:
        atoms_df (pd.DataFrame): DataFrame containing ALL atoms for the structure.
        chain_atoms_df (pd.DataFrame): DataFrame containing atoms for the SPECIFIC chain of the target residue.
        resid_tuple (tuple): (resname_letter, resnum) for the target residue.
        env_radius (float): Radius for neighbor search.
        base_transform_cpu (BaseTransform): An instantiated BaseTransform object (or compatible callable).
        return_debug_info (bool): If True, returns (graph, center, res_df), otherwise just graph.

    Returns:
        torch_geometric.data.Data or tuple or None: The generated graph or debug info, or None on failure.
    """
    resname_letter, resnum = resid_tuple
    protein_id = atoms_df['id'].iloc[0] if 'id' in atoms_df and not atoms_df.empty else 'unknown_protein'
    target_chain = chain_atoms_df['chain'].iloc[0] if 'chain' in chain_atoms_df and not chain_atoms_df.empty else None

    # --- Determine res_df (for center calculation and potential return) ---
    # Filter the chain-specific DataFrame to get atoms of the target residue
    res_df = pd.DataFrame() # Initialize
    if target_chain is not None and 'chain' in chain_atoms_df.columns and 'residue' in chain_atoms_df.columns:
        res_mask = (chain_atoms_df['chain'] == target_chain) & (chain_atoms_df['residue'] == resnum)
        res_df = chain_atoms_df.loc[res_mask]
    else:
         # This case should ideally not happen if chain_atoms_df is prepared correctly
         print(f"Warning: Could not determine target chain or missing columns for {protein_id}, residue {resnum}")

    if res_df.empty:
        # print(f"Warning: Residue {resname_letter}{resnum} (Chain: {target_chain}) not found in chain_atoms_df for {protein_id}.")
        return (None, None, None) if return_debug_info else None

    # --- Calculate Center ---
    center = sample_functional_center(res_df, resid_tuple, train_mode=False)
    if center is None:
        # print(f"Warning: Could not calculate center for {resname_letter}{resnum} in {protein_id}.")
        # Return res_df if requested, even if center calculation fails
        return (None, None, res_df) if return_debug_info else None

    # --- Build KDTree on FULL atoms_df and Find Neighbors ---
    # Use a copy to avoid modifying the original DataFrame passed to the function
    full_df_copy = atoms_df.copy()
    # Reset index *before* KDTree to match original logic's indexing basis
    full_df_copy = full_df_copy.reset_index()

    try:
        # Ensure coordinate columns exist in the full DataFrame
        if not {'x', 'y', 'z'}.issubset(full_df_copy.columns):
             print(f"Warning: Coordinate columns missing in full atoms_df for {protein_id}")
             return (None, center, res_df) if return_debug_info else None

        coords_full_np = full_df_copy[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        if coords_full_np.shape[0] == 0:
            print(f"Warning: Empty coordinates in full_df_copy for {protein_id}.")
            return (None, center, res_df) if return_debug_info else None

        # Use cKDTree to match original implementation
        tree = scipy.spatial.cKDTree(coords_full_np)
        pt_idx_list = tree.query_ball_point(center, r=env_radius) # Returns list of indices

        # query_ball_point returns list; check if it's empty
        if not pt_idx_list:
             # print(f"Warning: No points found within radius {env_radius} for center of {resname_letter}{resnum} in {protein_id}.")
             return (None, center, res_df) if return_debug_info else None

        # Indices refer to the reset index of full_df_copy
        # No need to convert to numpy array if iloc handles list directly
        pt_indices = pt_idx_list

        if not pt_indices: # Check again after potential conversion/filtering if any was added
             return (None, center, res_df) if return_debug_info else None

        # --- Create Environment DataFrame and Graph ---
        # Select rows using iloc on the FULL df copy, then reset index
        # .iloc handles the list of indices directly
        df_env = full_df_copy.iloc[pt_indices].reset_index(drop=True)

        if df_env.empty:
             # This shouldn't happen if pt_indices was not empty, but check defensively
             return (None, center, res_df) if return_debug_info else None

        # Ensure 'id' is present for BaseTransform, adding it if missing
        if 'id' not in df_env.columns:
            df_env['id'] = protein_id # Add protein ID

        # Call the provided BaseTransform instance
        graph = base_transform_cpu(df_env)

        if graph is not None:
             # Assign metadata to graph
             graph.protein_id = protein_id # Add protein ID to graph obj
             graph.resid = f"{resname_letter}{resnum}"
             graph.chain = target_chain if target_chain is not None else '?'
             graph.resname_letter = resname_letter # Store the letter used for center calc
             graph.resnum = resnum

        # --- Return Results ---
        if return_debug_info:
            return (graph, center, res_df)
        else:
            return graph

    except Exception as e:
        print(f"Error during KDTree/Graph creation for {resname_letter}{resnum} in {protein_id}: {e}")
        import traceback; traceback.print_exc()
        # Return res_df if available and debug info requested
        return (None, center, res_df) if return_debug_info and center is not None else ((None, None, None) if return_debug_info else None)


# =================== Parallel Processing Framework ===================

def prepare_graphs_for_protein_cpu(atom_df, include_hets, env_radius, base_transform_cpu):
    """
    Prepares list of graphs and metadata for a single protein (CPU Task).
    Iterates through residues of a protein and calls extract_env_for_residue_cpu.

    Args:
        atom_df (pd.DataFrame): Preprocessed DataFrame for a SINGLE protein (filtered, H removed etc.).
        include_hets (bool): Flag (though filtering now happens before this function).
        env_radius (float): Environment radius for graph extraction.
        base_transform_cpu (BaseTransform): Instantiated BaseTransform object.

    Returns:
        tuple[list, list]: List of generated graphs and list of corresponding metadata dicts.
    """
    graphs = []
    metadata = []
    protein_id_str = atom_df['id'].iloc[0] if 'id' in atom_df and not atom_df.empty else 'unknown_prep_cpu'

    # Basic check for required columns needed for iteration and extraction
    required_cols = ['chain', 'residue', 'resname', 'bfactor', 'element', 'x', 'y', 'z', 'name', 'id']
    if not all(col in atom_df.columns for col in required_cols):
        print(f"Warning: Missing required columns in atom_df for {protein_id_str} in prepare_graphs_for_protein_cpu.")
        return [], []

    # Map resname to single letter for iteration logic consistency
    try:
        if not (hasattr(atom_info, 'aa_to_letter') and callable(atom_info.aa_to_letter)):
             raise AttributeError("atom_info.aa_to_letter function missing")
        # Avoid modifying input df if possible, create series
        resname_letters_series = atom_df['resname'].apply(atom_info.aa_to_letter)
    except Exception as e:
        print(f"Error mapping resname to letter for {protein_id_str}: {e}")
        return [], []

    # Get unique standard amino acid residues to iterate over
    # Combine with original df to ensure alignment
    temp_df_for_iteration = atom_df[['chain', 'residue']].copy()
    temp_df_for_iteration['resname_letter'] = resname_letters_series
    residue_info_df = temp_df_for_iteration[['chain', 'residue', 'resname_letter']].drop_duplicates()

    # Filter for standard AA letters (use atom_info.aa_abbr if available)
    standard_letters = set()
    if hasattr(atom_info, 'aa_abbr'):
         standard_letters = set(atom_info.aa_abbr) - {'X'} # Exclude non-standard 'X' if present
    else:
         # Fallback if aa_abbr is missing (less robust)
         standard_letters = set('ACDEFGHIKLMNPQRSTVWY')
         print("Warning: atom_info.aa_abbr not found, using hardcoded standard AA letters.")

    standard_aa_mask = residue_info_df['resname_letter'].isin(standard_letters)
    residue_info_df = residue_info_df[standard_aa_mask]

    if residue_info_df.empty:
        # print(f"No standard residues found to process for {protein_id_str}") # Debug
        return [], []

    # --- Pre-group by chain for efficiency ---
    # Group the original atom_df once
    grouped_by_chain = {chain_id: group_df for chain_id, group_df in atom_df.groupby('chain')}
    # --- Pre-extract bfactors for lookup ---
    bfactor_df = atom_df[['chain', 'residue', 'bfactor']].drop_duplicates(subset=['chain', 'residue'])
    res_to_bfactor = dict(zip(zip(bfactor_df['chain'], bfactor_df['residue']), bfactor_df['bfactor']))
    # ----------------------------------------


    # Iterate through unique residues found
    for _, row in residue_info_df.iterrows():
        chain_id = row['chain']
        resnum = row['residue']
        resname_letter = row['resname_letter']

        # Get the pre-grouped DataFrame for this chain
        chain_atoms_df = grouped_by_chain.get(chain_id)
        if chain_atoms_df is None: # Should not happen if residue_info_df is derived correctly
            continue

        try:
            resnum_int = int(resnum)
        except ValueError:
            print(f"Warning: Non-integer residue number skipped: {resnum} for {protein_id_str}")
            continue

        resid_tuple = (resname_letter, resnum_int)

        # Call the core extraction function
        # Pass the full atom_df and the specific chain_atoms_df
        graph = extract_env_for_residue_cpu(
            atoms_df=atom_df, # Pass the full df for KDTree
            chain_atoms_df=chain_atoms_df, # Pass the chain df for center calc etc.
            resid_tuple=resid_tuple,
            env_radius=env_radius,
            base_transform_cpu=base_transform_cpu,
            return_debug_info=False # We only need the graph here
        )

        if graph is not None:
            graphs.append(graph)
            # Lookup bfactor (use 0.0 if not found)
            bfactor = res_to_bfactor.get((chain_id, resnum_int), 0.0)
            metadata.append({
                'protein_id': protein_id_str,
                'chain': chain_id,
                'resid': graph.resid, # Get resid from graph object if assigned
                'confidence': bfactor
            })

    return graphs, metadata

class GraphPreparationTransformCPU:
    """
    A callable class designed to be used as a transform in a DataLoader worker.
    It performs CPU-heavy preprocessing (filtering) and graph creation
    by calling prepare_graphs_for_protein_cpu.
    It uses the current BaseTransform (based on torch_cluster).
    """
    def __init__(self, include_hets=True, env_radius=10.0, num_rbf=16):
        self.include_hets = include_hets
        self.env_radius = env_radius
        self.num_rbf = num_rbf
        # BaseTransform is instantiated here, potentially within each worker process.
        # Ensure device is 'cpu' for worker-based execution.
        self.base_transform_cpu = BaseTransform(edge_cutoff=self.env_radius,
                                                num_rbf=self.num_rbf,
                                                device='cpu') # Explicitly CPU

    def __call__(self, elem):
        """ Processes one raw element (e.g., dict from ATOM3D dataset) """
        atom_df_raw = elem.get('atoms')
        protein_id = elem.get('id', 'unknown_protein')
        label = elem.get('label') # Preserve label if present

        if atom_df_raw is None or not isinstance(atom_df_raw, pd.DataFrame) or atom_df_raw.empty:
             # print(f"Worker skipping {protein_id}: No valid 'atoms' DataFrame.") # Debug
             return None

        # --- Apply Preprocessing ---
        try:
            # 1. Filter for first model (if applicable)
            atom_df = first_model_filter(atom_df_raw)
            # 2. Remove Water
            atom_df = atom_df[~atom_df.hetero.str.contains('W', na=False)]
            # 3. Remove Hydrogens
            atom_df = atom_df[atom_df['element'] != 'H']
            # 4. Filter Heteroatoms if requested
            if not self.include_hets:
                if hasattr(atom_info, 'aa') and isinstance(atom_info.aa, (list, set)):
                     if 'resname' in atom_df.columns:
                         atom_df = atom_df[atom_df.resname.isin(atom_info.aa)]
                     else:
                          print(f"Warning: 'resname' column missing for het filtering {protein_id}")
                          return None # Cannot filter hets without resname
                else:
                     print("Warning: atom_info.aa not found or not list/set, cannot filter hets.")
                     pass # Continue without het filtering if info is missing

            # 5. Reset index and check if empty
            atom_df = atom_df.reset_index(drop=True)
            if atom_df.empty:
                # print(f"Worker skipping {protein_id}: DataFrame empty after filtering.") # Debug
                return None

            # 6. Ensure ID column exists
            atom_df['id'] = protein_id

            # 7. Check essential columns exist AFTER filtering
            essential_cols = {'resname', 'chain', 'residue', 'element', 'x', 'y', 'z', 'name', 'bfactor', 'id'}
            if not essential_cols.issubset(atom_df.columns):
                 print(f"Worker skipping {protein_id}: Missing essential columns after filtering: {essential_cols - set(atom_df.columns)}")
                 return None

        except Exception as e:
            print(f"Worker skipping {protein_id} due to preprocessing error: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed trace
            return None
        # --- End Preprocessing ---

        # --- Prepare Graphs ---
        try:
            graphs, metadata = prepare_graphs_for_protein_cpu(
                atom_df=atom_df, # Pass the filtered df
                include_hets=self.include_hets, # Pass flag (though filtering done above)
                env_radius=self.env_radius,
                base_transform_cpu=self.base_transform_cpu # Pass the instantiated transform
            )

            if not graphs:
                # print(f"Worker: No graphs generated for {protein_id}.") # Debug
                return None

            # Return dict format expected by collate function
            result = {'graphs': graphs, 'metadata': metadata, 'id': protein_id}
            if label is not None:
                 result['label'] = label # Keep label if it existed
            return result

        except Exception as e:
             print(f"Worker error during graph preparation for {protein_id}: {e}")
             import traceback; traceback.print_exc() # Print detailed trace for this error
             return None
        # --- End Graph Preparation ---

class TransformedDatasetWrapper(Dataset):
    """
    Wraps a base dataset (e.g., ATOM3D LMDB dataset) and applies
    a transform (like GraphPreparationTransformCPU) in __getitem__.
    Intended for use with DataLoader multiprocessing.
    """
    def __init__(self, base_dataset, transform_cpu):
        """
        Args:
            base_dataset: The underlying dataset (e.g., loaded from LMDB).
            transform_cpu: An instance of the transform to apply (e.g., GraphPreparationTransformCPU).
        """
        if not hasattr(base_dataset, '__len__') or not hasattr(base_dataset, '__getitem__'):
             raise TypeError("base_dataset must support __len__ and __getitem__")
        if not callable(transform_cpu):
             raise TypeError("transform_cpu must be callable")

        self.base_dataset = base_dataset
        self.transform = transform_cpu # Store the transform instance

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """Loads raw item and applies the transform."""
        try:
            raw_item = self.base_dataset[idx]
            # Handle cases where the base dataset might return None
            if raw_item is None:
                 # print(f"Debug: Base dataset returned None for index {idx}") # Optional debug
                 return None
        except IndexError:
             print(f"Error: Index {idx} out of bounds for base dataset.")
             return None # Or re-raise? Returning None might be safer for DataLoader.
        except Exception as load_e:
            # Log error loading from base dataset
            print(f"Error loading RAW item {idx} from base dataset: {load_e}")
            return None

        # Apply the stored transform instance to the raw item
        try:
            transformed_item = self.transform(raw_item)
            # transform should return None on failure
            return transformed_item
        except Exception as transform_e:
            # Catch unexpected errors during the transform call itself
            protein_id = raw_item.get('id', f'index_{idx}') if isinstance(raw_item, dict) else f'index_{idx}'
            print(f"Unexpected error during transform application for {protein_id} (idx {idx}): {transform_e}")
            import traceback; traceback.print_exc()
            return None

# =================== Collate Function ===================

def graph_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Filters out None items (failed transforms) and batches valid graph data.
    Expects input `batch` to be a list of outputs from GraphPreparationTransformCPU
    (dicts containing 'graphs', 'metadata', 'id', optionally 'label').
    """
    # 1. Filter out None items (representing failed transformations)
    valid_items = [item for item in batch if isinstance(item, dict)]

    # If the entire batch failed
    if not valid_items:
        return None # Signal to the training loop to skip this batch

    # 2. Aggregate graphs and metadata from valid items
    all_graphs = []
    all_metadata = []
    original_ids = []
    labels = [] # Store labels if present
    has_labels = 'label' in valid_items[0] # Check if first item has label

    for item in valid_items:
        # Basic validation of item structure
        if 'graphs' in item and 'metadata' in item and item['graphs']:
            all_graphs.extend(item['graphs'])
            all_metadata.extend(item['metadata'])
            original_ids.append(item.get('id', 'unknown'))
            if has_labels:
                 labels.append(item.get('label')) # Append label, potentially None if missing
        else:
            # Log if a non-None item has unexpected structure
            item_id = item.get('id', 'unknown')
            # print(f"Warning: Collate received invalid item structure for {item_id}. Keys: {item.keys()}")

    # If aggregation resulted in no graphs (e.g., all valid items had empty graph lists)
    if not all_graphs:
        return None

    # 3. Create a single large batch graph using PyG's Batch
    try:
        final_graph_batch = Batch.from_data_list(all_graphs)
    except Exception as e:
        # Catch potential errors during batching (e.g., inconsistent Data objects)
        print(f"Error during Batch.from_data_list: {e}")
        # Try to identify which proteins might have caused the issue
        problematic_ids = list(set(m.get('protein_id', item_id) for item_id, m_list in zip(original_ids, all_metadata) for m in m_list))
        print(f"Potentially problematic IDs in batch leading to collation error: {problematic_ids}")
        return None # Signal failure for this batch

    # 4. Return the batch graph and aggregated metadata (and labels if applicable)
    if has_labels:
         # Convert labels to tensor or appropriate format if needed
         # Handle potential Nones if some items lacked labels
         # Example: return final_graph_batch, all_metadata, torch.tensor(labels) if all(l is not None for l in labels) else labels
         return final_graph_batch, all_metadata, labels # Returning list of labels for flexibility
    else:
         return final_graph_batch, all_metadata 