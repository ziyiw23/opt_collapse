import numpy as np
import os
import argparse
import torch
import pandas as pd
from atom3d.datasets import load_dataset, make_lmdb_dataset
import atom3d.util.file as fi
from collapse import initialize_model
import collections as col
import torch_cluster
from torch_geometric.data.data import Data
import random
from collapse.data import atom_info
from collapse.data import BaseTransform
# from collapse.data import sample_functional_center
from scipy.spatial import cKDTree

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = BaseTransform()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def extract_env_from_resid_heuristic(df, ch_resid, k_atoms=32, res_df = None, ca_center=False, train_mode=False):
    """
    Heuristic extraction: selects k closest heavy atoms to the residue center.
    """
    chain, resid = ch_resid
    if resid[0] == 'X':
        return None

    if res_df is None:
        df = df[df['chain'] == chain]
        df = df.loc[df['element'] != 'H']
        df['resname'] = df['resname'].map(aa_to_letter_dict).fillna('X')
        res_df = df[(df['resname'] == resid[0]) & (df['residue'] == int(resid[1:]))]
    
    if res_df.empty:
        return None

    if ca_center:
        try:
            center = res_df[res_df['name'] == 'CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
        except:
            return None
    else:
        center = sample_functional_center(res_df, resid, train_mode)

    coords_all = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=device)
    center_tensor = torch.as_tensor(center, dtype=torch.float32, device=device)
    dists = torch.norm(coords_all - center_tensor, dim=1)
    
    k_atoms = min(k_atoms, len(coords_all))
    tree = cKDTree(coords_all)
    distances, closest_idx = tree.query(center, k=k_atoms)
    # _, closest_idx = torch.topk(dists, k=k_atoms, largest=False)
    closest_idx = closest_idx.cpu().numpy()
    df_env = df.iloc[closest_idx].reset_index(drop=True)

    return transform(df_env)


def opt_embed_protein(atom_df, model, device='cuda', include_hets=True, env_radius=10.0, k_atoms=32):
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
        
        out = extract_env_from_resid_heuristic(chain_atoms, (chain, resid), k_atoms=k_atoms, ca_center=False, train_mode=False)
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
    def __init__(self, model, include_hets=True, k_atoms=32, device='cuda'):
        self.model = model
        self.include_hets = include_hets
        self.k_atoms = k_atoms
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
                                     include_hets=self.include_hets,
                                     k_atoms=self.k_atoms)
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
    parser.add_argument('--k_atoms', type=int, required=True, help="Number of closest atoms to consider for extraction of each residue environment.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(args.checkpoint, device=device)
    model.eval()
    transform = OptEmbedTransform(model, k_atoms=args.k_atoms, device=device)

    start_time = time.time()
    all_files = fi.find_files(args.data_dir, args.filetype)
    valid_files = [f for f in all_files if is_valid_pdb(f)]
    end_time = time.time()
    prefilter_time = end_time - start_time
    print(f"✅ Pre-filtering completed in {prefilter_time:.2f} seconds.")
    
    if len(valid_files) == 0:
        print("No valid PDB files found. Exiting...")
        return

    print(f"Processing {len(valid_files)} valid PDB files out of {len(all_files)} total.")

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

    make_lmdb_dataset(dataset, out_path, serialization_format='pkl', filter_fn=lambda x: x is None)

if __name__ == '__main__':
    main()