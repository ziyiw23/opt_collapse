import numpy as np
import os
import argparse
import torch
import pandas as pd
from atom3d.datasets import load_dataset, make_lmdb_dataset
import atom3d.util.file as fi
# Assuming collapse and atom_info are correctly importable
from collapse import initialize_model, atom_info
from atom3d.filters.filters import first_model_filter
from collapse.data import embed_residue
import collections as col
import random
import torch_cluster
from torch_geometric.data import Batch, Data
import torch_scatter
import time
import sys
import multiprocessing

def init_worker(df_global):
    """Initializer for multiprocessing pool workers."""
    global global_atom_df
    global_atom_df = df_global

# Define this function globally
def process_residue_center(residue_info_tuple):
    """
    Worker function to calculate center for a single residue.
    Accesses global_atom_df initialized by init_worker.
    """
    chain, resnum, resname_letter = residue_info_tuple

    if resname_letter == 'X':
        return (residue_info_tuple, None) # Skip 'X' residues

    # Access the global DataFrame (read-only access is safe)
    if 'global_atom_df' not in globals():
         # Safety check in case initializer failed (shouldn't happen with Pool)
         print("Error: global_atom_df not initialized in worker.")
         return (residue_info_tuple, None)

    try:
        # Filter the global DataFrame for the current residue
        res_mask = (global_atom_df['chain'] == chain) & (global_atom_df['residue'] == resnum)
        res_df = global_atom_df[res_mask]

        if res_df.empty:
            # Don't print from worker unless debugging, return None
            return (residue_info_tuple, None)

        # Call the existing center calculation function
        center = sample_functional_center(res_df, (resname_letter, resnum), train_mode=False)

        # Return the original identifier tuple and the result (center or None)
        return (residue_info_tuple, center)

    except Exception as e:
        print(f"Error processing {residue_info_tuple} in worker: {e}")
        return (residue_info_tuple, None) # Return None on error

def LINE():
    return sys._getframe(1).f_lineno

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_element_mapping = lambda x: {
    'C': 0,
    'N': 1,
    'O': 2,
    'F': 3,
    'S': 4,
    'Cl': 5, 'CL': 5,
    'P': 6,
    'Se': 7, 'SE': 7,
    'Fe': 8, 'FE': 8,
    'Zn': 9, 'ZN': 9,
    'Ca': 10, 'CA': 10,
    'Mg': 11, 'MG': 11,
}.get(x, 12)

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
    rbf = _rbf(E_vectors.norm(dim=-1),
               D_max=D_max, D_count=num_rbf, device=device)
    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_s, edge_v

class BaseTransform:
    '''
    Transforms atomic coordinates in a DataFrame into a torch_geometric Data graph.
    Includes CUDA stream support.
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu', stream=None):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
        # Use provided stream or default stream if on GPU, else None for CPU
        if isinstance(stream, torch.cuda.Stream) and device != 'cpu':
             self.stream = stream
        elif device != 'cpu':
             self.stream = torch.cuda.current_stream()
        else:
            self.stream = None # No stream context needed for CPU

    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates.
        :return: `torch_geometric.data.Data` structure graph.
        '''
        # Use stream context only if stream exists
        context = torch.cuda.stream(self.stream) if self.stream else torch.no_grad()
        with context:
             with torch.no_grad():
                coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                       dtype=torch.float32, device=self.device)
                atoms = torch.tensor(df['element'].map(_element_mapping).fillna(0).values,
                                     dtype=torch.long, device=self.device)

                # TODO: Tune radius_graph parameters
                edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)
                edge_s, edge_v = _edge_features(coords, edge_index, D_max=self.edge_cutoff,
                                                num_rbf=self.num_rbf, device=self.device)
                data = Data(x=coords, atoms=atoms,
                            edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)

                if 'same_chain' in df.columns:
                    data.chain_ind = torch.tensor(df['same_chain'].values, dtype=torch.long, device=self.device)

                return data

# --- Helper Functions ---
def sample_functional_center(df, resid_tuple, train_mode=False):
    """
    Calculates the center of a residue based on its functional atoms.

    Parameters:
     - df: pandas.DataFrame containing atoms ONLY for the target residue.
     - resid_tuple: Tuple (resname_letter, resnum), e.g., ('A', 123).
     - train_mode: Boolean, if True selects one random atom, if False uses all.

    Returns:
     - np.array: Coordinates of the calculated center (shape [3,]).
     - None: If no suitable center atoms found or df is empty.
    """
    if df.empty:
        print(f"Warning: DataFrame provided to sample_functional_center is empty for {resid_tuple}. Returning None.")
        return None

    resname, resnum = resid_tuple
    func_atoms_options = atom_info.abbr_key_atom_dict.get(resname, [])

    func_atoms = []
    if not func_atoms_options:
        func_atoms = ['CA']
    elif train_mode:
        # Training mode: select one random atom from all functional groups
        all_func_atoms = [atom for sublist in func_atoms_options for atom in sublist]
        if not all_func_atoms:
            func_atoms = ['CA']
        else:
            chosen_atom = np.random.choice(all_func_atoms)
            func_atoms = [chosen_atom]
    else:
        # Inference mode: use all defined functional atoms
        func_atoms = [atom for sublist in func_atoms_options for atom in sublist]
        if not func_atoms:
            func_atoms = ['CA']

    coords_all = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
    names = df['name'].to_numpy()
    # Create mapping {atom_name: coordinate_array} for atoms present in this residue
    name_to_coord = {name: coord for name, coord in zip(names, coords_all)}
    func_coords = [name_to_coord[name] for name in func_atoms if name in name_to_coord]

    # If no specified functional atoms were found in the df, try falling back to CA
    if not func_coords and 'CA' in name_to_coord:
        # Ensure 'CA' wasn't already the only option in func_atoms list
        if 'CA' not in func_atoms:
            print(f"Warning: Defined functional atoms {func_atoms} not found for residue {resname}{resnum}. Using CA instead.")
        func_coords = [name_to_coord['CA']]
    elif not func_coords:
        # If absolutely no functional atoms or CA found (e.g., incomplete residue in PDB)
        # Get the 3-letter code just for a more informative message, if possible
        three_letter_code_for_msg = df['resname'].iloc[0] if 'resname' in df.columns and not df.empty else resname
        print(f"Warning: No suitable center atoms (functional: {func_atoms} or CA) found for residue {three_letter_code_for_msg} {resnum}. Cannot calculate center.")
        return None

    center = np.mean(func_coords, axis=0, dtype=np.float32)

    return center


# --- New Vectorized Environment Extraction ---
def batch_extract_env(atom_df, unique_residues, env_radius, device, transforms):
    """
    Vectorized extraction of residue environments using torch.cdist and CUDA streams.

    Parameters:
     - atom_df: pandas.DataFrame with all atoms for the protein.
     - unique_residues: List/array of tuples like (chain, resnum, resname_letter).
     - env_radius: Float, radius for neighbor search.
     - device: Torch device.
     - transforms: List of BaseTransform instances (one per stream).

    Returns:
     - graphs: List of torch_geometric.data.Data objects.
     - valid_residues: List of (chain, resnum, resname_letter) tuples corresponding to successful graphs.
    """
    centers = []
    valid_res_info = [] # Store (chain, resnum, resname_letter) for mapping later
    res_dfs = {} # Cache res_df for center calculation

    all_coords_np = atom_df[['x', 'y', 'z']].values.astype(np.float32)

    # 1. Calculate centers sequentially (potential future optimization point)
    for chain, resnum, resname_letter in unique_residues:
        if resname_letter == 'X':
             continue
        # Filter atom_df *correctly* for the current residue
        res_mask = (atom_df['chain'] == chain) & (atom_df['residue'] == resnum)
        res_df = atom_df[res_mask]

        if res_df.empty:
            print(f"Warning: No atoms found for {chain}-{resname_letter}{resnum}. Skipping.")
            continue

        center = sample_functional_center(res_df, (resname_letter, resnum), train_mode=False)

        if center is not None:
            centers.append(center)
            valid_res_info.append((chain, resnum, resname_letter))
        else:
            print(f"Failed to get center for {chain}-{resname_letter}{resnum}. Skipping.")


    if not centers:
        print("No valid residue centers found for this protein.")
        return [], []

    # 2. Batch distance calculation
    all_coords_gpu = torch.tensor(all_coords_np, dtype=torch.float32, device=device)
    centers_gpu = torch.tensor(np.array(centers), dtype=torch.float32, device=device) # Shape: (M, 3)

    # Compute all pairwise distances: (N_atoms, M_centers)
    dists = torch.cdist(all_coords_gpu, centers_gpu)
    neighbor_masks = dists < env_radius 

    # 3. Batch graph generation (using streams)
    graphs = []
    final_valid_residues = [] # Residues for which graph generation succeeded
    num_transforms = len(transforms)

    for i in range(centers_gpu.shape[0]): # Iterate through centers (M)
        center_index = i
        # Find indices of atoms neighboring this center
        neighbor_atom_indices_gpu = torch.where(neighbor_masks[:, center_index])[0]

        if neighbor_atom_indices_gpu.numel() == 0:
            res_info = valid_res_info[center_index]
            print(f'No environment atoms found for {res_info[0]}-{res_info[2]}{res_info[1]} within {env_radius}A. Skipping.')
            continue

        # Select the corresponding transform instance using round-robin
        current_transform = transforms[center_index % num_transforms]

        # Get CPU indices to slice the original DataFrame
        neighbor_atom_indices_cpu = neighbor_atom_indices_gpu.cpu().numpy()
        df_env = atom_df.iloc[neighbor_atom_indices_cpu].reset_index(drop=True) # Create env DataFrame

        # Create graph using the selected transform (and its associated stream)
        # The transform.__call__ method uses its assigned stream context internally
        try:
            graph = current_transform(df_env)
            if graph is not None:
                 graphs.append(graph)
                 final_valid_residues.append(valid_res_info[center_index]) # Add corresponding residue info
            else:
                 res_info = valid_res_info[center_index]
                 print(f"Transform returned None for {res_info[0]}-{res_info[2]}{res_info[1]}. Skipping.")

        except Exception as e:
            res_info = valid_res_info[center_index]
            print(f"Error creating graph for {res_info[0]}-{res_info[2]}{res_info[1]}: {e}. Skipping.")
            continue


    # Ensure all streams are synchronized
    if device != 'cpu':
        for stream in [t.stream for t in transforms if t.stream is not None]:
             stream.synchronize()

    return graphs, final_valid_residues

def opt_embed_protein(atom_df, model, device, include_hets, env_radius, transforms):
    """
    Optimized protein embedding using vectorized environment extraction and batch inference.

    Parameters:
    - atom_df: Filtered DataFrame for the protein.
    - model: Pre-trained COLLAPSE model.
    - device: Torch device.
    - include_hets: Boolean, whether to include HETATMs.
    - env_radius: Float, radius for environments.
    - transforms: List of BaseTransform instances for stream parallelism.

    Returns:
    - emb_data: Dictionary containing 'embeddings', 'resids', 'chains', 'confidence', or None.
    """
    emb_data = col.defaultdict(list)

    chains = atom_df['chain'].values
    residues = atom_df['residue'].values
    resnames = atom_df['resname'].values
    bfactors = atom_df['bfactor'].values

    resname_letters = [atom_info.aa_to_letter(r) for r in resnames]

    residue_info = pd.DataFrame({
        'chain': chains,
        'residue': residues,
        'resname_letter': resname_letters,
        'bfactor': bfactors
    })

    if not include_hets:
        residue_info = residue_info[atom_df['resname'].isin(atom_info.aa)]

    unique_res_df = residue_info[['chain', 'residue', 'resname_letter', 'bfactor']].drop_duplicates(subset=['chain', 'residue'])
    unique_residues_list = [tuple(row) for row in unique_res_df[['chain', 'residue', 'resname_letter']].values]
    res_to_bfactor = dict(zip(zip(unique_res_df['chain'], unique_res_df['residue']), unique_res_df['bfactor']))

    graphs, valid_residues = batch_extract_env(atom_df, unique_residues_list, env_radius, device, transforms)

    if not graphs:
        protein_id = atom_df.get('id', 'unknown')
        print(f"No graphs generated for protein {protein_id}. Skipping embedding.")
        return None

    for chain, resnum, resname_letter in valid_residues:
        resid = f"{resname_letter}{resnum}"
        emb_data['resids'].append(resid)
        bfactor = res_to_bfactor.get((chain, resnum), 0.0)
        emb_data['confidence'].append(bfactor)

    def get_protein_id(df):
        return df['id'].iloc[0] if not df.empty and 'id' in df else 'unknown_protein_id'

    embeddings_list = []
    batch_size = 128 # Or 64, 256, etc.
    num_graphs = len(graphs)

    def get_protein_id(df):
        return df['id'].iloc[0] if not df.empty and 'id' in df else 'unknown_protein_id'

    with torch.no_grad():
        for i in range(0, num_graphs, batch_size):
            graph_chunk = graphs[i : i + batch_size]
            if not graph_chunk: continue

            graphs_batch_chunk = Batch.from_data_list(graph_chunk).to(device)

            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    embs_chunk, _ = model.online_encoder(graphs_batch_chunk, return_projection=False)
                    embs_chunk = embs_chunk.float()
                embeddings_list.append(embs_chunk.cpu())

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    protein_id_str = get_protein_id(atom_df)
                    print(f"⚠️ OOM on protein {protein_id_str} even during mini-batching (batch size {batch_size}, chunk starting at {i}). Attempting fallback.")
                    torch.cuda.empty_cache()
                    pooled_emb_np = pool_resid_embed(atom_df, valid_residues, model,
                                                     device=device, include_hets=include_hets,
                                                     env_radius=env_radius)

                    if pooled_emb_np is None:
                        print(f"❌ Fallback embedding also failed for {protein_id_str}")
                        return None 
                    else:
                        print(f"❌ Fallback provides protein-level embedding. Cannot return per-residue for {protein_id_str} due to mini-batch OOM.")
                        return None 

                else:
                    raise e

            # Clear cache more frequently if memory is extremely tight
            # torch.cuda.empty_cache()
    if not embeddings_list:
        protein_id_str = get_protein_id(atom_df)
        print(f"Warning: No embeddings generated after mini-batching for {protein_id_str}. Returning None.")
        return None # Essential check: if no embeddings, return None

    if not embeddings_list:
        print(f"Warning: No embeddings generated after mini-batching for {get_protein_id(atom_df)}.")
        return None

    final_embs = torch.cat(embeddings_list, dim=0)
    # ---- END MINI-BATCHING ----

    if isinstance(final_embs, torch.Tensor):
        emb_data['embeddings'] = final_embs.numpy()
    else:
        print(f"Warning: Final embeddings are not a tensor ({type(final_embs)}). Setting to None.")
        emb_data['embeddings'] = None
        return None

    if len(emb_data['embeddings']) != len(emb_data['resids']):
        protein_id_str_warn = get_protein_id(atom_df)
        print(f"Warning: Mismatch between final number of embeddings ({len(emb_data['embeddings'])}) and residues ({len(emb_data['resids'])}) for protein {protein_id_str_warn}.")
        return None

    return emb_data

def pool_resid_embed(atom_df, residue_list, model, device='cuda', include_hets=True, env_radius=10.0):
    """
    Fallback: Embeds residues sequentially using embed_residue and pools them.

    Parameters:
     - atom_df (pd.DataFrame): DataFrame with all protein atoms.
     - residue_list (List[Tuple[str, int, str]]): List of (chain, resnum, resname_letter) tuples
                                                  for residues that need embedding.
     - model (nn.Module): The embedding model.
     - device (str or torch.device): Device for computation.
     - include_hets (bool): Flag passed to embed_residue.
     - env_radius (float): Radius passed to embed_residue.

    Returns:
     - np.ndarray: Pooled protein-level embedding, or None if all residues fail.
    """
    residue_embs_list = []

    if 'embed_residue' not in globals():
         print("Error: Fallback function `embed_residue` is not defined or imported.")
         return None

    for chain, resnum, resname_letter in residue_list: 
        try:

            resid_str = f"{resname_letter}{resnum}"
            chain_resid_formatted = (chain, resid_str)

            emb_np = embed_residue(atom_df=atom_df,
                                   chain_resid=chain_resid_formatted,
                                   model=model,
                                   device=device,
                                   include_hets=include_hets,
                                   env_radius=env_radius)

            if emb_np is not None:
                if np.isfinite(emb_np).all():
                    residue_embs_list.append(emb_np)
                else:
                    protein_id_str = atom_df['id'].iloc[0] if not atom_df.empty and 'id' in atom_df else 'unknown'
                    print(f"⚠️ Fallback embed_residue for {protein_id_str} {chain_resid_formatted} resulted in non-finite values. Skipping.")

        except Exception as e:
            protein_id_str = atom_df['id'].iloc[0] if not atom_df.empty and 'id' in atom_df else 'unknown'
            print(f"⚠️ Fallback failed for {protein_id_str} attempting to embed residue with chain_resid={chain_resid_formatted}: {e}")
            # Print traceback for debugging
            # import traceback
            # traceback.print_exc()
            continue

    if not residue_embs_list:
        print("❌ Fallback: No residues successfully embedded.")
        return None

    try:
        out_all = torch.tensor(np.stack(residue_embs_list), dtype=torch.float32, device=device)
    except ValueError as e:
        print(f"❌ Fallback: Error stacking residue embeddings. Check for shape consistency. {e}")
        return None

    batch_id = torch.zeros(out_all.size(0), dtype=torch.long, device=out_all.device)

    protein_emb = None
    with torch.no_grad():
        if getattr(model, "scatter_mean", False):
            protein_emb = torch_scatter.scatter_mean(out_all, batch_id, dim=0)
        elif getattr(model, "attn", False) and hasattr(model, "global_attn"):
             protein_emb = torch.tanh(model.global_attn(out_all, batch_id))
        else:
             print("Warning: Model type for pooling (scatter_mean or attn) not specified in fallback. Using simple mean.")
             protein_emb = torch.mean(out_all, dim=0, keepdim=True)

    if protein_emb is None:
        return None

    if isinstance(protein_emb, torch.Tensor):
        return protein_emb.squeeze().cpu().numpy()
    else:
        print("Warning: Pooled embedding is not a tensor.")
        return None

# --- Modified OptEmbedTransform ---
class OptEmbedTransform(object):
    """
    A transform that applies the optimized embedding procedure to each dataset entry.
    Manages CUDA streams and multiple BaseTransform instances internally.
    """
    def __init__(self, model, include_hets=True, env_radius=10.0, device='cuda', num_streams=4):
        self.model = model
        self.include_hets = include_hets
        self.env_radius = env_radius
        self.device = device
        self.num_streams = num_streams if device != 'cpu' else 1

        # Initialize streams and transforms only if using CUDA
        if self.device != 'cpu':
            self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_streams)]
            self.transforms = [BaseTransform(edge_cutoff=self.env_radius, 
                                             num_rbf=16,
                                             device=self.device,
                                             stream=s) for s in self.streams]
        else:
            self.streams = [None]
            self.transforms = [BaseTransform(edge_cutoff=self.env_radius, num_rbf=16, device=self.device, stream=None)]


    def __call__(self, elem):
        atom_df_raw = elem['atoms']
        protein_id = elem.get('id', 'unknown')

        try:
            atom_df = first_model_filter(atom_df_raw)
            atom_df = atom_df[~atom_df.hetero.str.contains('W')] # Remove water
            atom_df = atom_df[atom_df.element != 'H'].reset_index(drop=True) # Remove hydrogens
            if not self.include_hets:
                atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)

            if atom_df.empty:
                 print(f"Skipping {protein_id}: No atoms left after filtering.")
                 return None

            atom_df['id'] = protein_id

            if 'resname' not in atom_df.columns:
                 print(f"Error: 'resname' column missing in DataFrame for {protein_id}. Cannot proceed.")
                 return None


        except Exception as e:
            print(f"Skipping {protein_id} due to preprocessing error: {e}")
            return None

        outdata = opt_embed_protein(atom_df, self.model, device=self.device,
                                      include_hets=self.include_hets,
                                      env_radius=self.env_radius,
                                      transforms=self.transforms)

        if outdata is None:
            print(f"Embedding failed for {protein_id}.")
            return None

        # Update the original element dictionary
        elem['resids'] = outdata['resids']
        elem['confidence'] = outdata['confidence']
        elem['chains'] = outdata['chains']
        elem['embeddings'] = outdata['embeddings']

        return elem

def is_valid_pdb(filepath):
    """Check if the file is non-empty before processing."""
    try:
        return os.path.getsize(filepath) > 0
    except OSError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Optimized embedding generation using vectorization and streams")
    parser.add_argument('data_dir', type=str, help="Directory containing PDB/mmCIF files or LMDB dataset")
    parser.add_argument('out_dir', type=str, help="Output directory for LMDB dataset")
    parser.add_argument('--split_id', type=int, default=0, help="Split ID (1 to num_splits) for processing subset")
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt', help="Path to model checkpoint")
    parser.add_argument('--filetype', type=str, default='pdb', help="Input file type (e.g., pdb, cif, pdb.gz, lmdb)")
    parser.add_argument('--num_splits', type=int, default=1, help="Number of splits to divide the dataset into")
    parser.add_argument('--num_streams', type=int, default=4, help="Number of CUDA streams for parallel graph construction")
    parser.add_argument('--env_radius', type=float, default=10.0, help="Radius for local environment extraction")
    parser.add_argument('--include_hets', action='store_true', help="Include HETATMs in processing")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed output on first few entries')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = initialize_model(args.checkpoint, device=device)
    model.eval()

    opt_transform = OptEmbedTransform(model,
                                    include_hets=args.include_hets,
                                    env_radius=args.env_radius,
                                    device=device,
                                    num_streams=args.num_streams)

    print(f"Loading dataset from: {args.data_dir} with filetype: {args.filetype}")
    dataset = load_dataset(args.data_dir, args.filetype, transform=opt_transform)

    print(f"Dataset size: {len(dataset)}")

    if args.num_splits > 1:
        if args.split_id < 1 or args.split_id > args.num_splits:
             raise ValueError(f"split_id must be between 1 and {args.num_splits}")
        try:
             indices = np.arange(len(dataset))
             split_indices = np.array_split(indices, args.num_splits)[args.split_id - 1]
             print(f'Processing split {args.split_id}/{args.num_splits} with {len(split_indices)} examples...')
             dataset = torch.utils.data.Subset(dataset, split_indices)
        except TypeError:
             print("Warning: Dataset does not support direct indexing for splitting. Processing full dataset.")
             out_path = args.out_dir
    else:
        print(f'Processing full dataset with {len(dataset)} examples...')
        out_path = args.out_dir

    if args.num_splits > 1:
        out_path = os.path.join(args.out_dir, f'tmp_split_{args.split_id}')
    else:
        out_path = args.out_dir

    os.makedirs(out_path, exist_ok=True)
    print(f"Output will be written to: {out_path}")


    if args.debug:
        print("🔍 Debug mode: inspecting first few entries after transform...")
        count = 0
        for i, item in enumerate(dataset):
            if count >= 5: break
            if item is None:
                print(f"[{i}] Skipped (transform returned None)")
                continue

            protein_id = item.get('id', f'unknown_{i}')
            print(f"--- Entry {i} | ID: {protein_id} ---")
            try:
                if 'embeddings' in item and item['embeddings'] is not None:
                    print(f"  ✅ Embeddings shape: {item['embeddings'].shape}")
                    print(f"  Residues found: {len(item.get('resids', []))}")
                else:
                    print(f"  ❌ No embeddings found for {protein_id}")

            except Exception as e:
                 print(f"  💥 Error inspecting item {protein_id}: {e}")

            count += 1
            print("-" * 20)
    else:
        print("Starting LMDB dataset creation...")
        start_time = time.time()
        make_lmdb_dataset(dataset, out_path,
                          serialization_format='pkl',
                          filter_fn=lambda x: x is None)
        end_time = time.time()
        print(f"LMDB dataset creation finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()