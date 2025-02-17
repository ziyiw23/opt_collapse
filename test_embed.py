import numpy as np
import os
import argparse
import torch
from collapse.data import EmbedTransform
from atom3d.datasets import load_dataset, make_lmdb_dataset
import atom3d.util.file as fi
from collapse import initialize_model
import collections as col
from collapse.data import atom_info
from collapse.data import BaseTransform
from collapse.data import sample_functional_center

transform = BaseTransform()

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
    import pandas as pd

    chain, resid = ch_resid
    if resid[0] == 'X':
        return None

    if res_df is None:
        df = df.copy()
        df['resname'] = df['resname'].apply(atom_info.aa_to_letter)
        rows = (df['chain'] == chain) & (df['resname'] == resid[0]) & (df['residue'] == int(resid[1:]))
        res_df = df[rows]

    if ca_center:
        try:
            center = res_df[res_df['name'] == 'CA'][['x','y','z']].astype(np.float32).to_numpy()[0]
        except Exception:
            return None
    else:
        center = sample_functional_center(res_df, resid, train_mode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy().astype(np.float32), device=device)
    center_tensor = torch.as_tensor(center, dtype=torch.float32, device=device)
    
    dists = torch.norm(coords - center_tensor, dim=1)
    neighbor_mask = dists < env_radius
    neighbor_indices = torch.where(neighbor_mask)[0]
    
    if neighbor_indices.numel() == 0:
        print('No environment found for', ch_resid)
        return None

    df_env = df.iloc[neighbor_indices.cpu().numpy(), :]

    graph = transform(df_env)
    return graph

def opt_embed_protein(atom_df, model, device='cuda', include_hets=True, env_radius=10.0):
    """
    Optimized embedding function that processes the input atom DataFrame,
    groups residues, and extracts local environment graphs using the optimized
    extract_env_from_resid. This version uses vectorized and batched operations.
    """
    import pandas as pd
    import collections as col
    from torch_geometric.data import Batch
    
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
        if out:
            graphs.append(out)
            emb_data['chains'].append(chain)
            emb_data['resids'].append(resid)
            emb_data['confidence'].append(bfactors[res_mask][0])
    
    if len(graphs) == 0:
        return None

    graphs = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        try:
            embs, _ = model.online_encoder(graphs, return_projection=False)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                print('Out of Memory error!', flush=True)
                return None
            raise e
    emb_data['embeddings'] = np.stack(embs.cpu().numpy(), 0)
    return emb_data

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
        from collapse.data import first_model_filter, atom_info
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

def main():
    import os
    import argparse
    from atom3d.datasets import load_dataset, make_lmdb_dataset
    import atom3d.util.file as fi
    from collapse import initialize_model

    parser = argparse.ArgumentParser(description="Optimized embedding generation for single-chain proteins")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
    parser.add_argument('--filetype', type=str, default='pdb')
    parser.add_argument('--num_splits', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(args.checkpoint, device=device)
    transform = OptEmbedTransform(model, device=device)

    dataset = load_dataset(args.data_dir, args.filetype, transform=transform)

    if args.num_splits > 1:
        out_path = os.path.join(args.out_dir, f'tmp_{args.split_id}')
        os.makedirs(out_path, exist_ok=True)
        import atom3d.util.file as fi
        all_files = fi.find_files(args.data_dir, args.filetype)
        split_idx = np.array_split(np.arange(len(all_files)), args.num_splits)[args.split_id - 1]
        print(f'Processing split {args.split_id} with {len(split_idx)} examples...')
        dataset = torch.utils.data.Subset(dataset, split_idx)
    else:
        out_path = args.out_dir
        print(f'Processing full dataset with {len(dataset)} examples...')

    make_lmdb_dataset(dataset, out_path, serialization_format='pkl', filter_fn=lambda x: (x is None))

if __name__ == '__main__':
    main()