# embedding_utils.py

import numpy as np
import torch
import pandas as pd
from scipy.spatial import KDTree
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
import collections as col
import random # May be needed if train_mode=True is ever used in sample_functional_center

# Assuming atom_info is importable and has the required structure
# If atom_info itself is complex, it might need careful handling
from collapse import atom_info

# Constants moved from gen_embed.py
ELEMENT_MAPPING = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'CL': 5,
    'P': 6, 'Se': 7, 'SE': 7, 'Fe': 8, 'FE': 8, 'Zn': 9, 'ZN': 9,
    'Ca': 10, 'CA': 10, 'Mg': 11, 'MG': 11,
}
DEFAULT_ELEMENT = 12

# --- Helper Functions (_normalize, _rbf, _edge_features) ---
def _normalize(tensor, dim=-1):
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    distances = torch.norm(E_vectors, dim=-1)
    rbf = _rbf(distances, D_max=D_max, D_count=num_rbf, device=device)
    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_s, edge_v

# --- BaseTransform (CPU Version for Workers) ---
class BaseTransform:
    """ Creates graph from DataFrame subset (now in utils file) """
    def __init__(self, edge_cutoff=4.5, num_rbf=16, max_neighbors=32, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.max_neighbors = max_neighbors
        self.device = device # Although intended for CPU, keep flexible

    def __call__(self, df):
        """ Creates graph """
        protein_id = df['id'].iloc[0] if 'id' in df and not df.empty else 'graph_gen'
        try:
            with torch.no_grad():
                elements_mapped = df['element'].map(ELEMENT_MAPPING)
                atoms = torch.as_tensor(elements_mapped.fillna(DEFAULT_ELEMENT).values,
                                      dtype=torch.long, device=self.device)
                coords_np = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                coords = torch.as_tensor(coords_np, dtype=torch.float32, device=self.device)

                tree = KDTree(coords_np)
                neighbors = tree.query_ball_point(coords_np, r=self.edge_cutoff)
                
                if self.max_neighbors > 0:
                    for i in range(len(neighbors)):
                        if len(neighbors[i]) > self.max_neighbors:
                            dists = np.linalg.norm(coords_np[neighbors[i]] - coords_np[i].reshape(1, -1), axis=1)
                            closest_indices = np.argsort(dists)[:self.max_neighbors]
                            neighbors[i] = [neighbors[i][j] for j in closest_indices]
                
                edge_list = []
                for i, neighbor_indices in enumerate(neighbors):
                    for j in neighbor_indices:
                        if i != j:
                            edge_list.append([i, j])
                
                if not edge_list:
                    return None
                    
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
                edge_s, edge_v = _edge_features(coords, edge_index, D_max=self.edge_cutoff,
                                               num_rbf=self.num_rbf, device=self.device)
                data = Data(x=coords, atoms=atoms,
                           edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
                return data
        except Exception as e:
            # print(f"Error during BaseTransform for {protein_id}: {e}") # Less verbose
            return None

# --- sample_functional_center ---
def sample_functional_center(res_df, resid_tuple, train_mode=False):
    if res_df.empty: return None
    resname_letter, resnum = resid_tuple
    func_atoms_options = atom_info.abbr_key_atom_dict.get(resname_letter, [])
    func_atoms = []
    if not func_atoms_options: func_atoms = ['CA']
    elif not train_mode:
        func_atoms = [atom for sublist in func_atoms_options for atom in sublist]
        if not func_atoms: func_atoms = ['CA']
    else: # train_mode=True
        all_func_atoms = [atom for sublist in func_atoms_options for atom in sublist]
        if not all_func_atoms: func_atoms = ['CA']
        else: func_atoms = [np.random.choice(all_func_atoms)]
    try:
        coords_all = res_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        names = res_df['name'].to_numpy()
        name_to_coord = {name: coord for name, coord in zip(names, coords_all)}
        func_coords = [name_to_coord[name] for name in func_atoms if name in name_to_coord]
        if not func_coords and 'CA' in name_to_coord:
            func_coords = [name_to_coord['CA']]
        elif not func_coords:
            return None
        center = np.mean(func_coords, axis=0, dtype=np.float32)
        return center
    except Exception as e:
        # print(f"Error in sample_functional_center for {resname_letter}{resnum}: {e}")
        return None

# --- extract_env_for_residue (CPU version using SciPy KDTree) ---
def extract_env_for_residue_cpu(chain_atoms_df, resid_tuple, env_radius, base_transform_cpu):
    """ Creates graph for one residue env using BaseTransformCPU. Runs on CPU."""
    resname_letter, resnum = resid_tuple
    protein_id = chain_atoms_df['id'].iloc[0] if 'id' in chain_atoms_df else 'unknown_chain'

    res_mask = (chain_atoms_df['residue'] == resnum)
    res_df = chain_atoms_df.loc[res_mask]
    if res_df.empty: return None

    center = sample_functional_center(res_df, resid_tuple, train_mode=False)
    if center is None: return None

    try:
        coords_chain_np = chain_atoms_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        if coords_chain_np.shape[0] == 0: return None

        tree = KDTree(coords_chain_np)
        pt_idx_list = tree.query_ball_point(center, r=env_radius)
        if not pt_idx_list: return None

        pt_idx_cpu = np.array(pt_idx_list, dtype=np.int64)
        if len(pt_idx_cpu) == 0: return None

        df_env = chain_atoms_df.iloc[pt_idx_cpu].reset_index(drop=True)
        df_env['id'] = protein_id

        graph = base_transform_cpu(df_env)

        if graph is not None:
             graph.resid = f"{resname_letter}{resnum}"
             graph.chain = chain_atoms_df['chain'].iloc[0]
             graph.resname_letter = resname_letter
             graph.resnum = resnum
        return graph

    except Exception as e:
        # print(f"Error extracting env (CPU) for {resname_letter}{resnum} in {protein_id}: {e}")
        return None

# --- prepare_graphs_for_protein (Worker Task Helper - CPU) ---
def prepare_graphs_for_protein_cpu(atom_df, include_hets, env_radius, base_transform_cpu):
    """ Prepares list of graphs and metadata for a single protein (CPU Task) """
    graphs = []
    metadata = []
    protein_id_str = atom_df['id'].iloc[0] if 'id' in atom_df else 'unknown_prep_cpu'

    required_cols = ['chain', 'residue', 'resname', 'bfactor', 'element', 'x', 'y', 'z', 'name']
    if not all(col in atom_df.columns for col in required_cols): return [], []

    try:
        if not (hasattr(atom_info, 'aa_to_letter') and callable(atom_info.aa_to_letter)):
             raise AttributeError("atom_info.aa_to_letter lambda func missing")
        resname_letters = atom_df['resname'].apply(atom_info.aa_to_letter).to_list()
        atom_df['resname_letter'] = resname_letters
    except Exception as e: return [], []

    residue_info_df = atom_df[['chain', 'residue', 'resname_letter', 'bfactor']].drop_duplicates(subset=['chain', 'residue'])
    standard_letters = set(atom_info.aa_abbr) - {'X'}
    standard_aa_mask = residue_info_df['resname_letter'].isin(standard_letters)
    residue_info_df = residue_info_df[standard_aa_mask]
    if not include_hets: pass
    if residue_info_df.empty: return [], []

    res_to_bfactor = dict(zip(zip(residue_info_df['chain'], residue_info_df['residue']), residue_info_df['bfactor']))
    grouped_by_chain = atom_df.groupby('chain')

    for chain_id, chain_atoms_df in grouped_by_chain:
        unique_residues_this_chain = residue_info_df[residue_info_df['chain'] == chain_id][['resname_letter', 'residue']].values
        if len(unique_residues_this_chain) == 0: continue

        for resname_letter, resnum in unique_residues_this_chain:
            try: resnum_int = int(resnum)
            except ValueError: continue
            resid_tuple = (resname_letter, resnum_int)

            graph = extract_env_for_residue_cpu(chain_atoms_df, resid_tuple, env_radius, base_transform_cpu)

            if graph is not None:
                graphs.append(graph)
                bfactor = res_to_bfactor.get((chain_id, resnum_int), 0.0)
                metadata.append({
                    'protein_id': protein_id_str,
                    'chain': chain_id,
                    'resid': graph.resid,
                    'confidence': bfactor
                })

    return graphs, metadata

# --- Graph Preparation Transform (CPU Version for Workers) ---
# Now defined in its own module, should be picklable
class GraphPreparationTransformCPU:
    """ Performs CPU-heavy preprocessing and graph creation in workers """
    def __init__(self, include_hets=True, env_radius=10.0, max_neighbors=32):
        self.include_hets = include_hets
        self.env_radius = env_radius
        self.max_neighbors = max_neighbors
        # Instantiated within the worker when the Dataset Wrapper calls it
        # Or, ensure it's initialized properly in worker_init_fn if needed
        self.base_transform_cpu = BaseTransform(edge_cutoff=self.env_radius,
                                                    num_rbf=16,
                                                    max_neighbors=self.max_neighbors)

    def __call__(self, elem):
        """ Processes one raw element from the dataset """
        atom_df_raw = elem.get('atoms')
        protein_id = elem.get('id', 'unknown')
        if atom_df_raw is None: return None
        
        # Need first_model_filter here
        from atom3d.filters.filters import first_model_filter

        try:
            atom_df = first_model_filter(atom_df_raw)
            atom_df = atom_df[~atom_df.hetero.str.contains('W')]
            atom_df = atom_df[atom_df.element != 'H']
            if not self.include_hets:
                if hasattr(atom_info, 'aa'):
                     if 'resname' in atom_df.columns:
                         atom_df = atom_df[atom_df.resname.isin(atom_info.aa)]
                     else: return None
                else: pass
            atom_df = atom_df.reset_index(drop=True)
            if atom_df.empty: return None
            atom_df['id'] = protein_id
            if not {'resname', 'chain', 'residue', 'element', 'x', 'y', 'z', 'name', 'bfactor'}.issubset(atom_df.columns):
                 return None
        except Exception as e:
            # print(f"Worker skipping {protein_id} due to preprocessing error: {e}")
            return None

        try:
            graphs, metadata = prepare_graphs_for_protein_cpu(atom_df,
                                                            self.include_hets,
                                                            self.env_radius,
                                                            self.base_transform_cpu)
            if not graphs: return None
            return {'graphs': graphs, 'metadata': metadata, 'id': protein_id}
        except Exception as e:
             # print(f"Worker error during graph preparation for {protein_id}: {e}")
             return None

# --- Dataset Wrapper (Applies CPU Transform in Worker) ---
# Now defined in its own module, should be picklable
class TransformedDatasetWrapper(Dataset):
    """ Wraps a base dataset and applies the CPU transform in __getitem__ """
    def __init__(self, base_dataset, transform_cpu):
        self.base_dataset = base_dataset
        # Store the transform *instance* (GraphPreparationTransformCPU)
        self.transform = transform_cpu 

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        try:
            raw_item = self.base_dataset[idx]
            if raw_item is None: return None
        except Exception as load_e:
            # print(f"Error loading RAW item {idx} from base dataset: {load_e}")
            return None

        try:
            # Apply the stored transform instance 
            transformed_item = self.transform(raw_item)
            return transformed_item
        except Exception as transform_e:
            protein_id = raw_item.get('id', f'index_{idx}') if raw_item else f'index_{idx}'
            # print(f"Unexpected error during transform call for {protein_id} in getitem: {transform_e}")
            return None 

# --- Custom Collate Function ---
def graph_collate_fn(batch):
    """ Collates processed items (dicts with graphs and metadata) from workers. """
    valid_items = []
    for idx, item in enumerate(batch):
        # Expecting dicts output by GraphPreparationTransformCPU
        if isinstance(item, dict) and 'graphs' in item and 'metadata' in item:
            # Ensure the list of graphs is not empty
            if item['graphs']: 
                valid_items.append(item)
        # Log if an item is None or not the expected format 
        # elif item is not None:
        #     print(f"Warning: Collate received unexpected item type: {type(item)} at index {idx}. Value: {str(item)[:100]}...")

    # If no valid items were found in the batch (e.g., all failed preprocessing)
    if not valid_items:
        return None, None # Signal to skip this batch

    # Combine graphs and metadata from all valid items in the batch
    all_graphs = []
    all_metadata = []
    for item in valid_items:
        all_graphs.extend(item['graphs']) # Assumes item['graphs'] is a list of Data objects
        all_metadata.extend(item['metadata']) # Assumes item['metadata'] is a list of dicts

    # If after combining, there are still no graphs (shouldn't happen if valid_items check passed)
    if not all_graphs:
        return None, None

    # Create a single large batch graph for efficient GPU processing
    try:
        # Batch.from_data_list handles the creation of the combined graph object
        final_graph_batch = Batch.from_data_list(all_graphs)
        return final_graph_batch, all_metadata
    except Exception as e:
        # Catch potential errors during batching (e.g., inconsistent Data objects)
        print(f"Error during Batch.from_data_list: {e}")
        # Try to identify which proteins might have caused the issue
        problematic_ids = list(set(m.get('protein_id', 'Unknown') for m in all_metadata))
        print(f"Potentially problematic IDs in batch leading to collation error: {problematic_ids}")
        return None, None # Signal failure for this batch 