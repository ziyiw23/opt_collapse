import argparse
import numpy as np
import pickle
import torch
from fastdist import fastdist
from atom3d.datasets import load_dataset
import collections as col
from collapse.utils import pdb_from_fname
from collapse import initialize_model
import re
from torch_geometric.data import Batch

# Import original transform
from collapse.data import EmbedTransform as OriginalEmbedTransform
from embedding_utils import GraphPreparationTransformCPU

parser = argparse.ArgumentParser()
parser.add_argument('pdb', type=str, nargs='+')
parser.add_argument('--mode', choices=['original', 'optimized'], required=True,
                    help='Embedding generation mode')
parser.add_argument('--chains', type=str, default=None)
parser.add_argument('--db', type=str, default='data/datasets/full_site_db_stats.pkl')
parser.add_argument('--cutoff', type=float, default=1e-4)
parser.add_argument('--site_cutoff', type=float, default=1e-4)
parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
parser.add_argument('--filetype', type=str, default='pdb')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--include_hets', action='store_true')
parser.add_argument('--debug', action='store_true', help='Run only on first 5 PDBs for quick debugging')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load database and model
with open(args.db, 'rb') as f:
    db = pickle.load(f)
db_embeddings = db['embeddings']
db_labels = np.array(db['sites'])
db_sources = np.array(db['sources'])
db_pdbs = np.array(db['pdbs'])
db_resids = np.array(db['resids'])
db_means = np.array(db['mean_cos'])
db_stds = np.array(db['std_cos'])
db_cutoffs = np.array([q[args.site_cutoff] for q in db['quantiles']])

# print(f'Using {args.mode} embedding mode')
# print(f'Searching {len(args.pdb)} PDBs against database of size {len(db_pdbs)}, representing {len(set(db_labels))} functional sites')

with open('data/background_stats/combined_background.pkl', 'rb') as f:
    quants = pickle.load(f)
    cutoff = quants[args.cutoff]
model = initialize_model(args.checkpoint, device=device)
model.eval()

if args.mode == 'original':
    transform = OriginalEmbedTransform(model, include_hets=args.include_hets, device=device)
else: # Optimized mode only prepares graphs here
    transform = GraphPreparationTransformCPU(include_hets=args.include_hets)

# Load dataset - transform is either OriginalEmbedTransform or GraphPreparationTransformCPU
dataset = load_dataset(args.pdb, args.filetype, transform=transform)

if args.debug:
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(min(5, len(dataset))))
    print("🔍 Debug mode: processing only the first 5 PDBs.")

db_pdbcodes = np.array([p[:4] for p in db_pdbs])

for data_from_loader in dataset:
    if data_from_loader is None:
        print("⚠️ Skipping failed transform/prep (None)")
        continue

    # ADDED: Conditional processing based on mode
    if args.mode == 'original':
        # Original mode output is already the final dictionary
        pdb_data = data_from_loader
    else: # Optimized mode - data_from_loader is the output of GraphPreparationTransformCPU
        processed_data = data_from_loader # Rename for clarity within this block
        if not isinstance(processed_data, dict) or 'graphs' not in processed_data or not processed_data.get('graphs'):
            pdb_id_err = processed_data.get('id', 'unknown') if isinstance(processed_data, dict) else 'unknown'
            print(f"⚠️ Skipping {pdb_id_err}: Invalid output from GraphPreparationTransformCPU.")
            continue

        graphs = processed_data['graphs']
        metadata = processed_data['metadata']
        pdb_id_from_prep = processed_data['id']

        if not graphs: # Redundant check, but safe
             print(f"⚠️ Skipping {pdb_id_from_prep}: No graphs generated.")
             continue

        # Batch graphs and run model
        embeddings_np = None # Initialize
        try:
            graph_batch_gpu = Batch.from_data_list(graphs).to(device)
            with torch.no_grad():
                 # Use autocast for potential performance gains on CUDA
                 with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                      # Assuming model.online_network exists and works like in gen_embed
                      embeddings_tensor = model.online_network(graph_batch_gpu)
                      embeddings_np = embeddings_tensor.float().cpu().numpy()

            del graph_batch_gpu # Clean up memory
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error during model inference for {pdb_id_from_prep}: {e}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Skip this protein

        if embeddings_np is None: # Check if inference failed
            print(f"❌ Skipping {pdb_id_from_prep} due to inference failure.")
            continue

        if len(metadata) != embeddings_np.shape[0]:
            print(f"❌ Mismatch between metadata ({len(metadata)}) and embeddings ({embeddings_np.shape[0]}) for {pdb_id_from_prep}. Skipping.")
            continue

        # Reconstruct the pdb_data dictionary
        resids_list = []
        chains_list = []
        confidence_list = []
        parsing_failed = False
        for i, meta in enumerate(metadata):
            match = re.match(r"([A-Za-z_]*)(\d+)", meta.get('resid', ''))
            if match:
                 resnum_str = match.group(2)
                 resids_list.append(resnum_str)
                 chains_list.append(meta.get('chain', '?'))
                 confidence_list.append(meta.get('confidence', 0.0))
            else:
                 print(f"⚠️ Warning: Could not parse resid '{meta.get('resid', '')}' for {pdb_id_from_prep}. Skipping protein.")
                 parsing_failed = True
                 break

        if parsing_failed:
            continue # Skip to the next protein

        if not resids_list: # Check if lists are empty after processing metadata
             print(f"⚠️ Skipping {pdb_id_from_prep}: No valid residues found after processing metadata.")
             continue

        pdb_data = {
            "id": pdb_id_from_prep,
            "resids": resids_list,
            "chains": chains_list,
            "embeddings": embeddings_np,
            "confidence": confidence_list
        }
    # --- End of conditional processing ---

    # --- Rest of the original loop logic starts here ---
    pdb_id, af_flag = pdb_from_fname(pdb_data["id"])
    print(f'Input PDB: {pdb_id}')

    if pdb_id[:4] in db_pdbcodes:
        idx_to_remove = np.where(db_pdbcodes == pdb_id[:4])[0]
        # Use boolean indexing for robust deletion
        keep_mask = np.ones(len(db_pdbs), dtype=bool)
        keep_mask[idx_to_remove] = False

        db_pdbs = db_pdbs[keep_mask]
        db_sources = db_sources[keep_mask]
        db_labels = db_labels[keep_mask]
        db_resids = db_resids[keep_mask]
        db_embeddings = db_embeddings[keep_mask]
        db_means = db_means[keep_mask]
        db_stds = db_stds[keep_mask]
        db_cutoffs = db_cutoffs[keep_mask]
        # Update db_pdbcodes as well after deletion
        db_pdbcodes = db_pdbcodes[keep_mask]

    resids = np.array(pdb_data['resids'])
    chains = np.array(pdb_data['chains'])
    embeddings = np.array(pdb_data['embeddings'])
    confidences = np.array(pdb_data['confidence'])

    if af_flag:
        # TODO: Verify if confidence filtering logic needs adjustment for 'optimized' mode (b-factors).
        print('Removing low confidence residues (threshold >= 70)')
        high_conf_idx = confidences >= 70
        if not np.any(high_conf_idx):
            print(f"Skipping {pdb_id} due to no high-confidence residues.")
            continue

        resids = resids[high_conf_idx]
        chains = chains[high_conf_idx]
        embeddings = embeddings[high_conf_idx]
        if embeddings.shape[0] == 0:
             print(f"Skipping {pdb_id} as no embeddings remained after confidence filtering.")
             continue

    if args.chains:
        print(f'Annotating chains: {args.chains}')
        chain_idx = np.in1d(chains, np.array(list(args.chains)))
        if not np.any(chain_idx):
            print(f"Skipping {pdb_id} due to no matching chains.")
            continue

        resids = resids[chain_idx]
        chains = chains[chain_idx]
        embeddings = embeddings[chain_idx]
        if embeddings.shape[0] == 0:
             print(f"Skipping {pdb_id} as no embeddings remained after chain filtering.")
             continue

    if embeddings.shape[0] == 0:
        print(f"Skipping {pdb_id} as no embeddings available for comparison.")
        continue
    if db_embeddings.shape[0] == 0:
        print(f"Skipping {pdb_id} as database is empty after potential self-removal.")
        continue

    cosines = fastdist.cosine_matrix_to_matrix(embeddings, db_embeddings)  # (n_res, n_db)

    query_mask = cosines > cutoff
    # Ensure broadcasting works correctly
    if db_cutoffs.ndim == 1 and query_mask.ndim == 2:
         site_mask = cosines > db_cutoffs[np.newaxis, :] # Ensure db_cutoffs is broadcast correctly
    else:
         print(f"Warning: Unexpected dimensions for cosine ({cosines.shape}) or db_cutoffs ({db_cutoffs.shape}). Skipping protein.")
         continue # Skip if dimensions are wrong

    quantile_mask = query_mask & site_mask

    results = col.defaultdict(dict)
    hit_idx_by_row = [np.nonzero(row)[0] for row in quantile_mask]
    
    for i, hit_idx in enumerate(hit_idx_by_row):
        if len(hit_idx) == 0:
            continue
        hits = np.unique(hit_idx)
        chain_res = chains[i] + '_' + resids[i]
        for h in hits:
            # Make sure index h is valid for DB arrays
            if h >= len(db_labels):
                 print(f"Warning: DB index {h} out of bounds for protein {pdb_id}. Skipping hit.")
                 continue

            key = (db_labels[h], db_sources[h])
            # Safely access DB info
            db_pdb_info = db_pdbs[h] if h < len(db_pdbs) else 'DB_PDB_?'
            db_resid_info = db_resids[h] if h < len(db_resids) else 'DB_RESID_?'
            hit_info_str = f'{db_pdb_info}: {db_resid_info}'

            if chain_res in results[key]:
                results[key][chain_res].add(hit_info_str)
            else:
                results[key][chain_res] = set([hit_info_str])
    
    print('Results at p = ', args.cutoff)
    if not results:
        print(" No significant hits found.")
    else:
        for (name, source), sites in results.items():
            print(f' {name} ({source})')
            for loc, pdbs in sites.items():
                if args.verbose:
                    print(f"    - {loc}: {pdbs}")
                else:
                    print(f"    - {loc}: {len(pdbs)} PDBs")