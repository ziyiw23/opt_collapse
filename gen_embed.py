import numpy as np
import os
import argparse
import torch
import pandas as pd
from atom3d.datasets import load_dataset, make_lmdb_dataset
import atom3d.util.file as fi
from collapse import initialize_model
import collections as col
import random
import torch_cluster
from torch_geometric.data import Batch, Data
from collapse.data import atom_info, first_model_filter, embed_residue
import torch_scatter


import sys

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
    '''
    Adapted from https://github.com/drorlab/gvp-pytorch
     
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    and 
    https://github.com/drorlab/gvp-pytorch
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
    
def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    """Adapted from https://github.com/drorlab/gvp-pytorch"""
    
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v

class BaseTransform:
    '''
    Adapted from https://github.com/drorlab/gvp-pytorch
    
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.
    
    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    
    Subclasses of BaseTransform will produce graphs with additional 
    attributes for the tasks-specific training labels, in addition 
    to the above.
    
    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.
    
    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
            
    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format
        
        :return: `torch_geometric.data.Data` structure graph
        '''
        with torch.no_grad():
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                     dtype=torch.float32, device=self.device)
            atoms = torch.tensor(df['element'].map(_element_mapping).fillna(0).values,
                     dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

            edge_s, edge_v = _edge_features(coords, edge_index, D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)
            
            data = Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
            
            if 'same_chain' in df.columns:
                data.chain_ind = torch.tensor(df['same_chain'].values, dtype=torch.long, device=self.device)
            
            return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = BaseTransform(device='cuda')

def sample_functional_center(df, resid, train_mode=True):
    func_atoms = atom_info.abbr_key_atom_dict[resid[0]]
    if train_mode:
        func_atoms = func_atoms[np.random.choice(len(func_atoms), replace=False)]
    else:
        func_atoms = sum(func_atoms, [])

    coords = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
    names = df['name'].to_numpy()
    name_to_coord = dict(zip(names, coords))
    func_coords = [name_to_coord[name] for name in func_atoms if name in name_to_coord]

    if len(func_coords) == 0 and 'CA' in name_to_coord:
        func_coords = [name_to_coord['CA']]

    center = np.mean(func_coords, axis=0)
    return center

def debug_print_atom_set(df_env, tag=""):
    """
    Print sorted list of atom tuples for debugging.
    Each tuple: (chain, residue, atom name, x, y, z)
    """
    atoms = df_env[['chain', 'residue', 'name', 'x', 'y', 'z']].to_numpy().tolist()
    atoms_sorted = [tuple(atom) for atom in atoms]
    print(f"{tag} Atom set (sorted):")
    for atom in atoms_sorted:
        print(atom)

def extract_env_from_resid(df, ch_resid, env_radius=10.0, res_df=None, ca_center=False, train_mode=False):
    """
    GPU-accelerated version using vectorized distance computation and torch_cluster.radius_graph.
    
    Parameters:
      - df: pandas.DataFrame with all atoms (columns: 'x','y','z', etc.)
      - ch_resid: tuple (chain, resid) where resid is like "A23"
      - env_radius: radius within which to search for neighbors
      - res_df: optional subset of df corresponding to the residue of interest
      - ca_center: if True, use CA atom as center; otherwise, use sample_functional_center()
      - train_mode: flag passed to sample_functional_center

    Returns:
      - graph: a torch_geometric.data.Data object representing the local environment
    """

    chain, resid = ch_resid
    if resid[0] == 'X':
        return None
     
    if res_df is None:
        df = df[df['chain'] == chain]
        df = df.loc[df['element'] != 'H']
        df['resname'] = df['resname'].map(aa_to_letter_dict).fillna('X')
        res_df = df[(df['resname'] == resid[0]) & (df['residue'] == int(resid[1:]))]
         
    
    if ca_center:
        try:
            center = res_df[res_df['name'] == 'CA'][['x','y','z']].astype(np.float32).to_numpy()[0]
        except Exception:
            return None
    else:
        center = sample_functional_center(res_df, resid, train_mode)

    coords = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32, device=device)
    center_tensor = torch.as_tensor(center, dtype=torch.float32, device=device)
    
    neighbor_mask = torch.norm(coords - center_tensor, dim=1) < env_radius
    neighbor_indices = torch.where(neighbor_mask)[0]
    
     
    if neighbor_indices.numel() == 0:
        print('No environment found for', ch_resid)
        return None

    pt_idx = neighbor_indices.cpu().numpy()
    df_env = df.iloc[pt_idx, :]
    graph = transform(df_env)
    return graph

def opt_embed_protein(atom_df, model, device='cuda', include_hets=True, env_radius=10.0):
    emb_data = col.defaultdict(list)
    graphs = []

    chains = atom_df['chain'].to_numpy()
    residues = atom_df['residue'].to_numpy()
    resnames = atom_df['resname'].to_numpy()
    bfactors = atom_df['bfactor'].to_numpy()

    residue_df = pd.DataFrame({'chain': chains, 'residue': residues, 'resname': resnames})
    mask = residue_df['resname'].isin(atom_info.aa[:20])
    if not include_hets:
        mask &= residue_df['resname'].isin(atom_info.aa)
    unique_residues = residue_df[mask].drop_duplicates().to_numpy()

    for chain, resnum, resname in unique_residues:
        res_mask = (chains == chain) & (residues == resnum)
        res_df = atom_df[res_mask]
        resid = f"{atom_info.aa_to_letter(resname)}{resnum}"
        chain_atoms = atom_df[atom_df.chain == chain]

        out = extract_env_from_resid(chain_atoms, (chain, resid), env_radius, res_df, ca_center=False, train_mode=False)
        if out is None:
            continue
        graphs.append(out)
        emb_data['chains'].append(chain)
        emb_data['resids'].append(resid)
        emb_data['confidence'].append(bfactors[res_mask][0])
        emb_data['chain_resids'].append((chain, resnum))  

    graphs = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        try:
            embs, _ = model.online_encoder(graphs, return_projection=False)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            protein_id = atom_df.get('id', 'unknown') if isinstance(atom_df, dict) else atom_df.get('id', 'unknown')
            print(f"⚠️  OOM on protein {protein_id}, retrying residue-wise")
            torch.cuda.empty_cache()
            embs = pool_resid_embed(atom_df, emb_data['chain_resids'], model,
                                    device=device, include_hets=include_hets, env_radius=env_radius)
            if embs is None:
                print(f"❌ Fallback embedding also failed for {protein_id}")
                return None

    if isinstance(embs, torch.Tensor):
        emb_data['embeddings'] = embs.cpu().numpy()
    else:
        emb_data['embeddings'] = embs[None, :]
    return emb_data


def pool_resid_embed(atom_df, residue_list, model, device='cuda', include_hets=True, env_radius=10.0):
    """
    Embeds and pools a list of residues to produce a protein-level embedding.

    Parameters:
        atom_df (pd.DataFrame): All atoms in the protein.
        residue_list (List[Tuple[str, str]]): List of (chain, resid) tuples to embed.
        model (nn.Module): COLLAPSE model with .online_encoder and .global_attn.
        device (str or torch.device): Device to run embeddings on.
        include_hets (bool): Whether to include heteroatoms in graphs.
        env_radius (float): Radius for local environment extraction.

    Returns:
        np.ndarray: Protein-level embedding, or None if all residues failed.
    """
    residue_embs = []

    for chain_resid in residue_list:
        try:
            emb = embed_residue(atom_df, chain_resid, model, device=device,
                                include_hets=include_hets, env_radius=env_radius)
            if emb is not None:
                residue_embs.append(emb)
        except Exception as e:
            print(f"⚠️ Failed to embed {chain_resid}: {e}")
            continue

    if not residue_embs:
        print("❌ No residues successfully embedded.")
        return None

    out_all = torch.tensor(np.stack(residue_embs), dtype=torch.float32, device=device)
    batch_id = torch.zeros(out_all.size(0), dtype=torch.long, device=out_all.device)

    # Apply pooling
    if getattr(model, "scatter_mean", False):
        protein_emb = torch_scatter.scatter_mean(out_all, batch_id, dim=0)

    elif getattr(model, "attn", False):
        protein_emb = torch.tanh(model.global_attn(out_all, batch_id))

    else:
        raise ValueError("Model must specify either scatter_mean or attn pooling")

    return protein_emb.squeeze().cpu().numpy()


class OptEmbedTransform(object):
    """
    A transform that applies the optimized embedding procedure to each dataset entry.
    """
    def __init__(self, model, include_hets=True, env_radius=10.0, device='cuda'):
        self.model = model
        self.include_hets = include_hets
        self.env_radius = env_radius
        self.device = device
    
    def __call__(self, elem):
        # print(f"Processing: {elem.get('id', 'unknown')}")
        atom_df = elem['atoms']
        try:  
            atom_df = first_model_filter(atom_df)  
            atom_df = atom_df[~atom_df.hetero.str.contains('W')]  
            atom_df = atom_df[atom_df.element != 'H'].reset_index(drop=True)  
            if not self.include_hets:    
                atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)   
        except Exception as e:
            print(f"Skipping {elem.get('id', 'unknown')}: {e}")
            return None
         
        outdata = opt_embed_protein(atom_df, self.model, device=self.device,
                                     include_hets=self.include_hets, env_radius=self.env_radius)
        if outdata is None:
            return None
        
        elem['resids'] = outdata['resids']
        elem['confidence'] = outdata['confidence']
        elem['chains'] = outdata['chains']
        elem['embeddings'] = outdata['embeddings']
        return elem

def is_valid_pdb(filepath):
    """Check if the file is non-empty before processing."""
    return os.path.getsize(filepath) > 0

def main():
    import time

    parser = argparse.ArgumentParser(description="Optimized embedding generation for single-chain proteins")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
    parser.add_argument('--filetype', type=str, default='pdb.gz')
    parser.add_argument('--num_splits', type=int, default=1)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(args.checkpoint, device=device)
    model.eval()
    transform = OptEmbedTransform(model, device=device)
     

    dataset = load_dataset(args.data_dir, args.filetype, transform=transform)

    if args.num_splits > 1:
        out_path = os.path.join(args.out_dir, f'tmp_{args.split_id}')
        os.makedirs(out_path, exist_ok=True)
        split_idx = np.array_split(np.arange(len(valid_files)), args.num_splits)[args.split_id - 1]
        print(f'Processing split {args.split_id} with {len(split_idx)} examples...')
        dataset = torch.utils.data.Subset(dataset, split_idx)
    else:
        out_path = args.out_dir
        print(f'Processing full dataset with {len(dataset)} examples...')

    if args.debug:
        print("🔍 Debug mode: inspecting first few PDB entries...")
        for i, item in enumerate(dataset):
            try:
                print(f"[{i}] ID: {item['id']}")
                out = transform(item)
                if out is None:
                    print(f"❌ Transform returned None for {item['id']}")
                else:
                    print(f"✅ Got embeddings for {item['id']} — shape: {out['embeddings'].shape}")
            except Exception as e:
                print(f"💥 Error processing {item['id']}: {e}")
            
            if i == 5:
                break
    else:
        make_lmdb_dataset(dataset, out_path, 
            serialization_format='pkl', 
            filter_fn=lambda x: (x is None))


if __name__ == '__main__':
    main()