# V2.4 - DataLoader Parallelism

import numpy as np
import os
import argparse
import torch
import pandas as pd
# Use atom3d.datasets.LMDBDataset if reading from LMDB, or parse files directly/use atom3d.dataset.FileDataset
from atom3d.datasets import load_dataset, make_lmdb_dataset# Check exact function if needed
import atom3d.util.file as fi
from collapse import initialize_model, atom_info # Assuming these are correct
from atom3d.filters.filters import first_model_filter
import collections as col
import random
import torch_cluster
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # Might be useful for metadata if needed
import lmdb # For manual LMDB writing
import pickle # For LMDB serialization
from tqdm import tqdm # Progress bar

import time
import sys

# --- Seeding and Constants ---
# (Keep Seeding and ELEMENT_MAPPING as before)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ELEMENT_MAPPING = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'CL': 5,
    'P': 6, 'Se': 7, 'SE': 7, 'Fe': 8, 'FE': 8, 'Zn': 9, 'ZN': 9,
    'Ca': 10, 'CA': 10, 'Mg': 11, 'MG': 11,
}
DEFAULT_ELEMENT = 12

# --- Helper Functions (_normalize, _rbf, _edge_features) ---
# (Keep these functions as before)
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

# --- BaseTransform (Used by workers) ---
# (Keep V2.3 version with as_tensor fix)
class BaseTransform:
    """ Creates graph from DataFrame subset """
    def __init__(self, edge_cutoff=4.5, num_rbf=16, max_neighbors=32, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.max_neighbors = max_neighbors
        # Force CPU device for graph creation in workers to avoid CUDA context issues
        self.device = torch.device('cpu') # Workers should use CPU for graph gen

    def __call__(self, df):
        """ Creates graph on CPU """
        protein_id = df['id'].iloc[0] if 'id' in df and not df.empty else 'graph_gen'
        try:
            with torch.no_grad():
                elements_mapped = df['element'].map(ELEMENT_MAPPING)
                if elements_mapped.isnull().any():
                    unknown_elements = df['element'][elements_mapped.isnull()].unique()
                    # print(f"Warning: Unknown elements {unknown_elements} in {protein_id}, mapping to {DEFAULT_ELEMENT}")
                elements_np = elements_mapped.fillna(DEFAULT_ELEMENT).values
                elements_np_int64 = elements_np.astype(np.int64)
                # Create atoms tensor on CPU
                atoms = torch.as_tensor(elements_np_int64, dtype=torch.long, device=self.device)

                coords_np = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                 # Create coords tensor on CPU
                coords = torch.as_tensor(coords_np, dtype=torch.float32, device=self.device)

                edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff,
                                                        max_num_neighbors=self.max_neighbors,
                                                        batch=None)

                edge_s, edge_v = _edge_features(coords, edge_index, D_max=self.edge_cutoff,
                                                num_rbf=self.num_rbf, device=self.device) # Features also on CPU

                data = Data(x=coords, atoms=atoms,
                            edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
                # Store protein_id within the data object for easy retrieval after batching
                data.protein_id = protein_id
                return data
        except Exception as e:
            print(f"Error during BaseTransform for {protein_id}: {e}")
            return None

# --- sample_functional_center ---
# (Keep V2.3 version)
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
        print(f"Error in sample_functional_center for {resname_letter}{resnum}: {e}")
        return None

# --- extract_env_for_residue (Helper for worker transform) ---
# Renamed from extract_env_from_resid, simplified
def extract_env_for_residue(chain_atoms_df, resid_tuple, env_radius, base_transform):
    """ Creates graph for one residue env using BaseTransform. Runs on CPU."""
    resname_letter, resnum = resid_tuple
    protein_id = chain_atoms_df['id'].iloc[0] if 'id' in chain_atoms_df else 'unknown_chain'

    res_mask = (chain_atoms_df['residue'] == resnum)
    res_df = chain_atoms_df.loc[res_mask]
    if res_df.empty: return None

    center = sample_functional_center(res_df, resid_tuple, train_mode=False)
    if center is None: return None

    try:
        # Neighbor search on CPU using NumPy/SciPy or simple loops if torch_cluster CPU is slow/unavailable
        # Option 1: Basic NumPy loop (might be slow for large chains)
        coords_chain_np = chain_atoms_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        dists_sq = np.sum((coords_chain_np - center)**2, axis=1)
        neighbor_mask = dists_sq < (env_radius**2)
        pt_idx_cpu = np.where(neighbor_mask)[0]

        # # Option 2: If you have scipy installed (often faster)
        # from scipy.spatial import KDTree
        # coords_chain_np = chain_atoms_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        # tree = KDTree(coords_chain_np)
        # pt_idx_cpu = tree.query_ball_point(center, r=env_radius)
        # if not pt_idx_cpu: return None # Convert list to numpy array if needed later

        if len(pt_idx_cpu) == 0: return None

        # Slice to get environment df
        df_env = chain_atoms_df.iloc[pt_idx_cpu].reset_index(drop=True)
        df_env['id'] = protein_id # Add protein ID for context

        # Create graph using BaseTransform (will be on CPU)
        graph = base_transform(df_env)
        # Add residue info to the graph object itself before returning
        if graph is not None:
             graph.resid = f"{resname_letter}{resnum}"
             graph.chain = chain_atoms_df['chain'].iloc[0] # Get chain from chain_df
             graph.resname_letter = resname_letter
             graph.resnum = resnum
        return graph

    except Exception as e:
        print(f"Error extracting environment (CPU) for {resname_letter}{resnum} in {protein_id}: {e}")
        return None

# --- prepare_graphs_for_protein (Worker Task Helper) ---
def prepare_graphs_for_protein(atom_df, include_hets, env_radius, base_transform):
    """ Prepares list of graphs and metadata for a single protein (CPU Task) """
    # This function runs within the DataLoader worker process

    graphs = []
    metadata = [] # Store metadata corresponding to each graph generated

    protein_id_str = atom_df['id'].iloc[0] if 'id' in atom_df else 'unknown_prep'

    # --- Prepare unique residues ---
    required_cols = ['chain', 'residue', 'resname', 'bfactor', 'element', 'x', 'y', 'z', 'name']
    if not all(col in atom_df.columns for col in required_cols):
        # print(f"Error: Missing required columns in DataFrame for {protein_id_str}")
        return [], [] # Return empty lists

    try:
        if not (hasattr(atom_info, 'aa_to_letter') and callable(atom_info.aa_to_letter)):
             raise AttributeError("atom_info.aa_to_letter lambda function not found or not callable")
        resname_letters = atom_df['resname'].apply(atom_info.aa_to_letter).to_list()
        atom_df['resname_letter'] = resname_letters
    except Exception as e:
         print(f"Error mapping resnames for {protein_id_str} in worker: {e}")
         return [], [] # Return empty lists

    residue_info_df = atom_df[['chain', 'residue', 'resname_letter', 'bfactor']].drop_duplicates(subset=['chain', 'residue'])
    standard_letters = set(atom_info.aa_abbr) - {'X'}
    standard_aa_mask = residue_info_df['resname_letter'].isin(standard_letters)
    residue_info_df = residue_info_df[standard_aa_mask]

    if not include_hets:
        pass

    if residue_info_df.empty:
         return [], []

    res_to_bfactor = dict(zip(zip(residue_info_df['chain'], residue_info_df['residue']), residue_info_df['bfactor']))

    grouped_by_chain = atom_df.groupby('chain')

    for chain_id, chain_atoms_df in grouped_by_chain:
        unique_residues_this_chain = residue_info_df[residue_info_df['chain'] == chain_id][['resname_letter', 'residue']].values
        if len(unique_residues_this_chain) == 0: continue

        for resname_letter, resnum in unique_residues_this_chain:
            try:
                resnum_int = int(resnum)
            except ValueError: continue # Skip if resnum isn't int
            resid_tuple = (resname_letter, resnum_int)

            # Pass the single transform instance created for this worker
            graph = extract_env_for_residue(chain_atoms_df, resid_tuple, env_radius, base_transform)

            if graph is not None:
                graphs.append(graph)
                # Get metadata associated with this graph
                bfactor = res_to_bfactor.get((chain_id, resnum_int), 0.0)
                metadata.append({
                    'protein_id': protein_id_str, # Changed from graph.protein_id
                    'chain': chain_id,
                    'resid': graph.resid, # Get info attached in extract_env
                    'confidence': bfactor
                })

    return graphs, metadata # Return list of graphs and list of metadata dicts

# --- Graph Preparation Transform (for DataLoader Workers) ---
class GraphPreparationTransform:
    """ Performs CPU-heavy preprocessing and graph creation in workers """
    def __init__(self, include_hets=True, env_radius=10.0, max_neighbors=32):
        self.include_hets = include_hets
        self.env_radius = env_radius
        # Create a BaseTransform instance FOR EACH WORKER when initialized
        # Important: Graph creation must happen on CPU within worker
        self.base_transform = BaseTransform(edge_cutoff=self.env_radius,
                                            num_rbf=16,
                                            max_neighbors=max_neighbors,
                                            device='cpu') # Ensure CPU

    def __call__(self, elem):
        """ Processes one raw element from the dataset """
        atom_df_raw = elem.get('atoms')
        protein_id = elem.get('id', 'unknown')

        if atom_df_raw is None:
             print(f"Warning: Worker received item with no 'atoms' key for {protein_id}")
             return None # Indicate failure

        try:
            # Preprocessing
            atom_df = first_model_filter(atom_df_raw)
            atom_df = atom_df[~atom_df.hetero.str.contains('W')]
            atom_df = atom_df[atom_df.element != 'H']
            if not self.include_hets:
                if hasattr(atom_info, 'aa'):
                     if 'resname' in atom_df.columns:
                         atom_df = atom_df[atom_df.resname.isin(atom_info.aa)]
                     else: return None # Cannot filter
                else: pass # atom_info missing 'aa'

            atom_df = atom_df.reset_index(drop=True)
            if atom_df.empty: return None
            atom_df['id'] = protein_id # Add ID back for prepare_graphs function
            if not {'resname', 'chain', 'residue', 'element', 'x', 'y', 'z', 'name', 'bfactor'}.issubset(atom_df.columns):
                 return None

        except Exception as e:
            print(f"Worker skipping {protein_id} due to preprocessing error: {e}")
            return None # Indicate failure

        # Call the graph preparation function
        try:
            graphs, metadata = prepare_graphs_for_protein(atom_df,
                                                        self.include_hets,
                                                        self.env_radius,
                                                        self.base_transform)
            if not graphs: # If prepare_graphs returned empty list
                return None
            return {'graphs': graphs, 'metadata': metadata}

        except Exception as e:
             print(f"Worker error during graph preparation for {protein_id}: {e}")
             # import traceback; traceback.print_exc() # Makes logs verbose
             return None # Indicate failure


# --- Custom Collate Function (More Robust)---
def graph_collate_fn(batch):
    """ Collates outputs from workers into a single large graph batch and metadata list """
    valid_items = []
    # Filter out None results and ensure items are dictionaries with required keys
    for idx, item in enumerate(batch):
        # Check if item is a dictionary and has the keys we need
        if isinstance(item, dict) and 'graphs' in item and 'metadata' in item:
            # Optionally, only include items that actually produced graphs
            if item['graphs']: # Check if the graphs list is not empty
                valid_items.append(item)
            # else: # Optional logging if needed
            #    print(f"Debug: Worker returned item with empty graphs list (Index {idx}).")
        elif item is not None:
            # Log unexpected non-None items that aren't the correct dict structure
            print(f"Warning: Collate received unexpected item type: {type(item)} at index {idx}. Value: {str(item)[:100]}...") # Print type and truncated value

    # If the batch is empty after filtering valid items
    if not valid_items:
        # print("Debug: Collate function resulted in an empty batch.") # Optional logging
        return None, None # Return None if batch is empty

    all_graphs = []
    all_metadata = []

    # Iterate through the validated items
    for item in valid_items:
        all_graphs.extend(item['graphs'])
        all_metadata.extend(item['metadata']) # Keep metadata as a flat list of dicts

    # Double-check if all_graphs is empty - shouldn't happen if valid_items check passed
    if not all_graphs:
        print("Warning: No graphs collected in collate function despite valid items.")
        return None, None

    try:
        # Create the single large batch for the GPU
        final_graph_batch = Batch.from_data_list(all_graphs)
        return final_graph_batch, all_metadata
    except Exception as e:
        # Catch errors during Batch.from_data_list, which can be sensitive
        print(f"Error during Batch.from_data_list: {e}")
        # Try to identify which proteins might have caused the issue
        problematic_ids = list(set(m.get('protein_id', 'Unknown') for m in all_metadata))
        print(f"Potentially problematic IDs in batch leading to collation error: {problematic_ids}")
        # import traceback; traceback.print_exc() # Uncomment for deeper debug
        return None, None # Indicate failure


# --- Dataset Wrapper ---
class Atom3DDatasetWrapper(Dataset):
    """ Wraps an existing atom3d dataset """
    def __init__(self, atom3d_dataset, transform=None):
        self.dataset = atom3d_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Return the raw item, transform will be applied by DataLoader worker
        try:
            item = self.dataset[idx]
            if self.transform is not None:
                item = self.transform(item)
            return item
        except Exception as e:
            print(f"Error loading item {idx} from base dataset: {e}")
            return None # Return None to be filtered by collate_fn

# --- is_valid_pdb ---
# (Keep as before)
def is_valid_pdb(filepath):
    try: return os.path.getsize(filepath) > 0
    except OSError: return False

def get_gpu_memory_info():
    """Get GPU memory information in human readable format"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)
        free = total_memory - allocated
        
        def bytes_to_gb(bytes):
            return bytes / (1024**3)
            
        return {
            'total': f"{bytes_to_gb(total_memory):.2f}GB",
            'allocated': f"{bytes_to_gb(allocated):.2f}GB",
            'cached': f"{bytes_to_gb(cached):.2f}GB",
            'free': f"{bytes_to_gb(free):.2f}GB"
        }
    return None

def get_batch_memory_usage(graph_batch):
    """Get memory usage of a batch in human readable format"""
    if graph_batch is None:
        return "0B"
    
    def get_tensor_memory(tensor):
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.nelement()
    
    total_memory = 0
    for key, value in graph_batch:
        if isinstance(value, torch.Tensor):
            total_memory += get_tensor_memory(value)
    
    # Convert to human readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_memory < 1024:
            return f"{total_memory:.2f}{unit}"
        total_memory /= 1024
    return f"{total_memory:.2f}TB"

def embed_residue_batch(graph_batch, model, device='cpu'):
    """Process a batch of residue graphs and return their embeddings"""
    if not isinstance(graph_batch, Batch) or len(graph_batch) == 0:
        return None
        
    graph_batch = graph_batch.to(device)
    with torch.no_grad():
        with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
            embs, _ = model.online_encoder(graph_batch, return_projection=False)
            return embs.float().cpu().numpy()

def find_optimal_chunk_size(graphs, model, device, start_size=100):
    """Binary search to find largest number of residues that fit in GPU memory"""
    left, right = 1, start_size
    max_size = 1
    
    while left <= right:
        mid = (left + right) // 2
        try:
            # Clear cache before test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try processing mid number of graphs
            test_batch = Batch.from_data_list(graphs[:mid]).to(device)
            test_embs = embed_residue_batch(test_batch, model, device)
            
            if test_embs is not None:
                max_size = mid
                left = mid + 1
            else:
                right = mid - 1
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                right = mid - 1
            else:
                raise e
            
    return max_size

class OptEmbedTransform:
    """Optimized transform that processes residues in optimal-sized chunks.
    This transform is specifically for use by annotate_pdb.py and similar single-protein processing tasks."""
    def __init__(self, model, include_hets=True, env_radius=10.0, device='cpu'):
        self.model = model
        self.include_hets = include_hets
        self.env_radius = env_radius
        self.device = device
        # Create transform instances
        self.graph_transform = GraphPreparationTransform(
            include_hets=include_hets,
            env_radius=env_radius
        )
        
    def __call__(self, elem):
        """Process one protein and return embeddings"""
        # First use the graph preparation transform to get graphs and metadata
        result = self.graph_transform(elem)
        if result is None:
            return None
            
        graphs, metadata = result['graphs'], result['metadata']
        if not graphs:
            return None
            
        try:
            # Find optimal chunk size for these graphs
            chunk_size = find_optimal_chunk_size(graphs, self.model, self.device)
            
            # Process in chunks
            all_embeddings = []
            for i in range(0, len(graphs), chunk_size):
                chunk = graphs[i:i + chunk_size]
                chunk_batch = Batch.from_data_list(chunk)
                chunk_embs = embed_residue_batch(chunk_batch, self.model, self.device)
                
                if chunk_embs is not None:
                    all_embeddings.extend(chunk_embs)
                else:
                    print(f"Warning: Failed to get embeddings for chunk {i//chunk_size}")
                    return None
                    
            if not all_embeddings:
                return None
                
            # Prepare output in the same format as before for compatibility with annotate_pdb.py
            embeddings = np.stack(all_embeddings, axis=0)
            return {
                'id': elem.get('id', 'unknown'),
                'embeddings': embeddings,
                'resids': [m['resid'] for m in metadata],
                'chains': [m['chain'] for m in metadata],
                'confidence': [m['confidence'] for m in metadata]
            }
            
        except Exception as e:
            print(f"Error in OptEmbedTransform: {e}")
            return None

# --- main function (V2.4 - DataLoader Parallelism) --- ## MODIFIED ##
def main():
    parser = argparse.ArgumentParser(description="V2.4 Embedding generation with DataLoader Parallelism")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
    parser.add_argument('--filetype', type=str, default='pdb')
    parser.add_argument('--num_splits', type=int, default=1)
    parser.add_argument('--env_radius', type=float, default=10.0)
    parser.add_argument('--include_hets', action='store_true', default=False)
    parser.add_argument('--max_neighbors', type=int, default=32, help="Max neighbors for radius graph (default 32)")
    parser.add_argument('--compile_model', action='store_true', help="Enable torch.compile (PyTorch 2.0+)")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers for parallel processing")
    parser.add_argument('--batch_size', type=int, default=1, help="Number of *proteins* per batch for DataLoader")  # Reduced default batch size
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print initial GPU memory info
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print("\nInitial GPU Memory Information:")
        for key, value in gpu_info.items():
            print(f"{key}: {value}")
    
    print(f"Using num_workers: {args.num_workers}, batch_size: {args.batch_size}")

    print("Loading model...")
    model = initialize_model(args.checkpoint, device=device)
    model.eval()

    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        try:
            model = torch.compile(model, mode="default")
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")

    print(f"Loading RAW dataset structure from: {args.data_dir} with filetype: {args.filetype}")
    try:
        raw_dataset = load_dataset(args.data_dir, args.filetype, transform=None)
        dataset_len = len(raw_dataset)
        print(f"Initial raw dataset size: {dataset_len}")
        if dataset_len == 0:
            print("Error: Loaded raw dataset is empty.")
            sys.exit(1)
        
        graph_transform = GraphPreparationTransform(include_hets=args.include_hets,
                                                env_radius=args.env_radius,
                                                max_neighbors=args.max_neighbors)
        dataset = Atom3DDatasetWrapper(raw_dataset, transform=graph_transform)
    except Exception as e:
        print(f"Error loading raw dataset: {e}")
        sys.exit(1)

    if args.num_splits > 1:
        if args.split_id < 1 or args.split_id > args.num_splits:
            print(f"Error: split_id ({args.split_id}) must be between 1 and {args.num_splits}")
            sys.exit(1)
        indices = np.arange(dataset_len)
        split_indices = np.array_split(indices, args.num_splits)[args.split_id - 1]
        if len(split_indices) == 0:
            print(f"Warning: Split {args.split_id} has 0 examples after splitting dataset of size {dataset_len}.")
        print(f'Processing split {args.split_id}/{args.num_splits} with {len(split_indices)} examples...')
        dataset = torch.utils.data.Subset(dataset, split_indices)
        out_path = os.path.join(args.out_dir, f'embeddings_split_{args.split_id}')
    else:
        print(f'Processing full dataset with {len(dataset)} examples...')
        out_path = args.out_dir

    os.makedirs(out_path, exist_ok=True)
    lmdb_path = os.path.join(out_path, 'data.lmdb')
    print(f"Output LMDB will be written to: {lmdb_path}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_collate_fn,
        pin_memory=True if str(device) != 'cpu' else False,
        worker_init_fn=lambda worker_id: random.seed(seed + worker_id)
    )

    print("Starting embedding generation loop...")
    start_time = time.time()
    results_to_save = []
    processed_count = 0
    failed_count = 0

    for batch_idx, (graph_batch, metadata_batch) in enumerate(tqdm(dataloader, desc="Processing Batches")):
        if graph_batch is None or metadata_batch is None:
            print(f"Warning: Skipping empty batch {batch_idx}")
            failed_count += args.batch_size
            continue

        try:
            # Print memory info before processing batch
            # if torch.cuda.is_available():
                # gpu_info = get_gpu_memory_info()
                # batch_memory = get_batch_memory_usage(graph_batch)
                # print(f"\nBatch {batch_idx} Memory Information:")
                # print(f"Number of residues: {len(metadata_batch)}")
                # print(f"Batch memory usage: {batch_memory}")
                # for key, value in gpu_info.items():
                    # print(f"{key}: {value}")

            # Clear CUDA cache before processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move graph batch to GPU
            graph_batch = graph_batch.to(device)

            with torch.no_grad():
                try:
                    with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                        embs, _ = model.online_encoder(graph_batch, return_projection=False)
                        final_embs = embs.float().cpu().numpy()
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"⚠️ OOM error during GNN inference on batch {batch_idx}. Processing in optimal-sized chunks...")
                        
                        # Get list of individual graphs from the batch
                        graphs = graph_batch.to_data_list()
                        total_residues = len(graphs)
                        
                        # Find optimal chunk size through binary search
                        chunk_size = find_optimal_chunk_size(graphs, model, device)
                        print(f"Found optimal chunk size: {chunk_size} residues")
                        
                        # Process all residues in chunks
                        final_embs = []
                        current_protein_results = col.defaultdict(lambda: col.defaultdict(list))
                        
                        for i in range(0, total_residues, chunk_size):
                            chunk = graphs[i:i + chunk_size]
                            chunk_meta = metadata_batch[i:i + chunk_size]
                            
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                                # Create batch from chunk and get embeddings
                                chunk_batch = Batch.from_data_list(chunk)
                                chunk_embs = embed_residue_batch(chunk_batch, model, device)
                                
                                if chunk_embs is not None:
                                    final_embs.extend(chunk_embs)
                                    # Process chunk results
                                    for j, meta in enumerate(chunk_meta):
                                        protein_id = meta['protein_id']
                                        current_protein_results[protein_id]['resids'].append(meta['resid'])
                                        current_protein_results[protein_id]['chains'].append(meta['chain'])
                                        current_protein_results[protein_id]['confidence'].append(meta['confidence'])
                                        processed_count += 1
                                else:
                                    print(f"Warning: Chunk {i//chunk_size} returned None embeddings")
                                    failed_count += len(chunk)
                                    
                            except Exception as e:
                                print(f"Error processing chunk {i//chunk_size}: {e}")
                                failed_count += len(chunk)
                                continue
                        
                        if not final_embs:
                            print(f"Warning: No embeddings generated for batch {batch_idx}")
                            failed_count += len(metadata_batch)
                            continue
                            
                        final_embs = np.stack(final_embs, axis=0)
                        
                    else:
                        raise e
                except Exception as e:
                    print(f"Non-runtime error during inference on batch {batch_idx}: {e}")
                    failed_count += len(list(set(m['protein_id'] for m in metadata_batch)))
                    continue

            if final_embs is None:
                print(f"Warning: Embeddings are None after inference for batch {batch_idx}. Skipping.")
                failed_count += len(list(set(m['protein_id'] for m in metadata_batch)))
                continue

            # Process batch results
            ptr = graph_batch.ptr.cpu().numpy()
            current_protein_results = col.defaultdict(lambda: col.defaultdict(list))

            if len(metadata_batch) != final_embs.shape[0]:
                print(f"CRITICAL WARNING: Mismatch after inference! Metadata length ({len(metadata_batch)}) != Embeddings length ({final_embs.shape[0]}) for batch {batch_idx}. Skipping batch.")
                failed_count += len(list(set(m['protein_id'] for m in metadata_batch)))
                continue

            for i, meta in enumerate(metadata_batch):
                protein_id = meta['protein_id']
                current_protein_results[protein_id]['resids'].append(meta['resid'])
                current_protein_results[protein_id]['chains'].append(meta['chain'])
                current_protein_results[protein_id]['confidence'].append(meta['confidence'])
                current_protein_results[protein_id]['embeddings'].append(final_embs[i])

            for protein_id, data in current_protein_results.items():
                if data['embeddings']:
                    data['embeddings'] = np.stack(data['embeddings'], axis=0)
                    if len(data['resids']) == data['embeddings'].shape[0]:
                        results_to_save.append({'id': protein_id, **data})
                        processed_count += 1
                    else:
                        print(f"Final internal mismatch for {protein_id}. Resids: {len(data['resids'])}, Embs: {data['embeddings'].shape[0]}. Skipping.")
                        failed_count += 1
                else:
                    print(f"No embeddings collected for {protein_id}. Skipping.")
                    failed_count += 1

        except Exception as e:
            print(f"Unexpected error processing batch {batch_idx}: {e}")
            failed_count += args.batch_size
            continue

    end_time = time.time()
    print(f"Embedding generation loop finished in {end_time - start_time:.2f} seconds.")
    print(f"Successfully processed: {processed_count} proteins.")
    print(f"Failed/Skipped: {failed_count} proteins (due to errors or OOM).")

    print(f"Writing {len(results_to_save)} results to LMDB: {lmdb_path}")
    if not results_to_save:
        print("No results to save.")
        return

    map_size = 1024 * 1024 * 1024 * 50  # 50 GB initial size

    try:
        env = lmdb.open(lmdb_path, map_size=map_size)
        with env.begin(write=True) as txn:
            for i, result_dict in enumerate(tqdm(results_to_save, desc="Writing LMDB")):
                key = result_dict['id'].encode('utf-8')
                value = pickle.dumps(result_dict)
                txn.put(key, value)
        env.close()
        print("LMDB writing complete.")
    except Exception as e:
        print(f"Error writing to LMDB: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()