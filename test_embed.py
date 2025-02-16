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
from torch_geometric.data import Batch
from tqdm import tqdm


def extract_env_from_resid(df, ch_resid, env_radius=10.0, res_df=None, ca_center=False, train_mode=False):
    """
    Optimized version: uses vectorized distance computation instead of a KD-tree.
    
    Parameters:
      - df: pandas.DataFrame with all atoms (must have columns 'x','y','z')
      - ch_resid: tuple (chain, resid) where resid is like "A23"
      - env_radius: radius to search for neighbors
      - res_df: optional, subset of df corresponding to the residue
      - ca_center: if True, use the CA atom for the center
      - train_mode: flag for training vs. inference (used in sample_functional_center)
      
    Returns:
      - graph: the transformed graph (using a global `transform` function)
    """
    from collapse.data import atom_info

    chain, resid = ch_resid
    if resid[0] == 'X':
        return None

    # If a residue-specific dataframe is not provided, filter df to get it.
    if res_df is None:
        df = df.copy()  # avoid modifying original DataFrame
        df['resname'] = df['resname'].apply(atom_info.aa_to_letter)
        rows = (df['chain'] == chain) & (df['resname'] == resid[0]) & (df['residue'] == int(resid[1:]))
        res_df = df[rows]

    # Determine the center point for the residue.
    if ca_center:
        try:
            center = res_df[res_df['name'] == 'CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
        except Exception:
            return None
    else:
        # sample_functional_center should be defined elsewhere.
        center = sample_functional_center(res_df, resid, train_mode)

    # --- Optimized: Vectorized Distance Calculation ---
    # Convert all atomic coordinates (for the entire chain) to a NumPy array.
    coords = df[['x', 'y', 'z']].to_numpy().astype(np.float32)
    # Convert to a torch tensor
    coords_tensor = torch.tensor(coords)  # you can move this to GPU if needed
    center_tensor = torch.tensor(center)
    # Compute Euclidean distances in a vectorized fashion.
    dists = torch.norm(coords_tensor - center_tensor, dim=1)
    # Find indices of atoms within env_radius.
    mask = dists < env_radius
    indices = mask.nonzero().squeeze().cpu().numpy()
    print(f"Center: {center}, Found {len(indices)} atoms within radius {env_radius}")
    if len(indices) == 0:
        print('No environment found')
        return None
    df_env = df.iloc[indices, :]
    
    # Convert the environment DataFrame into a graph.
    transform = BaseTransform()
    graph = transform(df_env)
    return graph

def opt_embed_protein(atom_df, model, device='cpu', include_hets=True, env_radius=10.0):
    emb_data = col.defaultdict(list)
    graphs = []
    if not include_hets:
        atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)
    for (c, i, r), res_df in atom_df.groupby(['chain', 'residue', 'resname']):
        if r not in atom_info.aa[:20]:
            continue
        emb_data['chains'].append(c)
        resid = atom_info.aa_to_letter(r) + str(i)
        chain_atoms = atom_df[atom_df.chain == c]
        out = extract_env_from_resid(chain_atoms, (c, resid), env_radius, res_df.copy(), train_mode=False)
        if out is None:
            continue
        graphs.append(out)
        emb_data['resids'].append(resid)
        confidence = res_df['bfactor'].iloc[0]  # for AlphaFold pLDDT
        emb_data['confidence'].append(confidence)
    graphs = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        try:
            embs, _ = model.online_encoder(graphs, return_projection=False)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Out of Memory error!', flush=True)
            return None
    emb_data['embeddings'] = np.stack(embs.cpu().numpy(), 0)
    return emb_data


class OptEmbedTransform(object):
    '''
    Transforms LMDB PDBDataset entries
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph
    '''
    
    def __init__(self, model, include_hets=True, env_radius=10.0, device='cpu'):
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
        except:
            return None

        outdata = opt_embed_protein(atom_df, self.model, device=self.device, include_hets=self.include_hets, env_radius=self.env_radius)
        if outdata is None:
            return
        elem['resids'] = outdata['resids']
        elem['confidence'] = outdata['confidence']
        elem['chains'] = outdata['chains']
        elem['embeddings'] = outdata['embeddings']
        return elem


def main():

    parser = argparse.ArgumentParser()
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

# e_db = []
# pdb_ids = []
# chains = []
# resids = []
# for elem in tqdm(dataset):
#     if elem is None:
#         continue
#     name, _ = pdb_from_fname(elem['id'])
#     resids.extend(elem['chains'])
#     resids.extend(elem['resids'])
#     pdb_ids.extend([name] * len(elem['resids']))
#     e_db.append(elem['embeddings'])

# e_db = np.stack(e_db, 0)

# print(e_db.shape)

# outdata = {'embeddings': e_db, 'pdbs': pdb_ids, 'chains': chains, 'resids': resids}

# with open(args.outfile, 'wb') as f:
#     pickle.dump(outdata, f)
