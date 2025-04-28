import argparse
import numpy as np
import pickle
import torch
from fastdist import fastdist
from atom3d.datasets import load_dataset
import collections as col
from collapse.utils import pdb_from_fname
from collapse import initialize_model

# Import original transform
from collapse.data import EmbedTransform as OriginalEmbedTransform
# Import components needed for the optimized path
from embedding_utils import GraphPreparationTransformCPU # The CPU preprocessing transform
from torch_geometric.data import Batch # To batch graphs for the model

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
# Add arguments to match gen_embed.py defaults, allowing override
parser.add_argument('--env_radius', type=float, default=10.0, help="Environment radius for graph construction")
parser.add_argument('--max_neighbors', type=int, default=32, help="Max neighbors for radius graph")

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

# Load raw dataset - transform will be applied conditionally inside the loop
dataset = load_dataset(args.pdb, args.filetype, transform=None) 

if args.debug:
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(min(5, len(dataset))))
    print("🔍 Debug mode: processing only the first 5 PDBs.")

db_pdbcodes = np.array([p[:4] for p in db_pdbs])

# After loading database
print("\nDatabase statistics:")
print(f"- Number of embeddings: {len(db_embeddings)}")
print(f"- Embedding dimension: {db_embeddings.shape[1]}")
print(f"- Number of unique sites: {len(set(db_labels))}")
print(f"- Cutoff value: {cutoff}")

for raw_pdb_data in dataset:
    if raw_pdb_data is None: 
        print("⚠️ Skipping entry: Raw data is None.")
        continue
        
    # Check if essential keys are present
    if 'id' not in raw_pdb_data or 'atoms' not in raw_pdb_data:
        print(f"⚠️ Skipping entry: Missing 'id' or 'atoms' key in raw data: {raw_pdb_data.get('id', 'Unknown ID')}")
        continue
        
    pdb_id, af_flag = pdb_from_fname(raw_pdb_data["id"])
    print(f'\nProcessing Input PDB: {pdb_id}')
    
    processed_pdb_data = None # Initialize variable to hold processed data
    
    # --- Conditional Processing based on Mode --- 
    if args.mode == 'original':
        print(f"  Using original embedding mode...")
        try:
            transform = OriginalEmbedTransform(model, include_hets=args.include_hets, device=device)
            processed_pdb_data = transform(raw_pdb_data)
            if processed_pdb_data is None:
                print(f"  ⚠️ Original transform failed for {pdb_id}.")
        except Exception as e:
             print(f"  ⚠️ Error during original transform for {pdb_id}: {e}")
             processed_pdb_data = None 
             
    elif args.mode == 'optimized':
        print(f"  Using optimized embedding mode (replicating gen_embed logic)...")
        try:
            # 1. Prepare Graphs on CPU using the transform from embedding_utils
            graph_transform_cpu = GraphPreparationTransformCPU(
                include_hets=args.include_hets, 
                env_radius=args.env_radius,      # Pass the argument
                max_neighbors=args.max_neighbors # Pass the argument
            )
            # graph_transform_cpu expects a dict like raw_pdb_data
            prepared_data = graph_transform_cpu(raw_pdb_data)

            if prepared_data is None or not prepared_data.get('graphs'):
                print(f"  ⚠️ Graph preparation failed or yielded no graphs for {pdb_id}.")
                processed_pdb_data = None
            else:
                graphs_cpu = prepared_data['graphs']
                metadata = prepared_data['metadata'] # Contains resids, chains, confidence per graph
                
                # 2. Batch graphs and run inference on the target device
                # print(f"    Generated {len(graphs_cpu)} graphs. Running inference...") # Debug
                graph_batch = Batch.from_data_list(graphs_cpu).to(device)
                
                with torch.no_grad():
                    # Use autocast for potential speedup/memory saving on GPU
                    with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                        embs_gpu, _ = model.online_encoder(graph_batch, return_projection=False)
                        # Ensure output is float32 for consistency
                        embeddings_np = embs_gpu.float().cpu().numpy()

                # 3. Verify and reconstruct the output dictionary
                if len(metadata) != embeddings_np.shape[0]:
                    print(f"  ⚠️ Metadata length ({len(metadata)}) mismatch with embeddings shape ({embeddings_np.shape}) for {pdb_id}.")
                    processed_pdb_data = None
                else:
                    processed_pdb_data = {
                        'id': pdb_id, # Use the cleaned pdb_id
                        'embeddings': embeddings_np,
                        'resids': [m['resid'] for m in metadata],
                        'chains': [m['chain'] for m in metadata],
                        'confidence': [m['confidence'] for m in metadata],
                        # Add original atoms or filepath if subsequent code needs them
                        # 'atoms': raw_pdb_data['atoms'], 
                        # 'file_path': raw_pdb_data.get('file_path') 
                    }
                    # print(f"    Successfully generated embeddings. Shape: {embeddings_np.shape}") # Debug

        except Exception as e:
            print(f"  ⚠️ Error during optimized processing for {pdb_id}: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed debug
            processed_pdb_data = None
            
    else: # Should not happen with choices defined in argparse
        print(f"  ⚠️ Unknown mode: {args.mode}")
        continue
        
    # --- Check if processing was successful ---    
    if processed_pdb_data is None:
        print(f"⚠️ Skipping {pdb_id} due to processing failure in mode '{args.mode}'.")
        continue
        
    # --- Continue with annotation using processed_pdb_data --- 
    # Remove self from database if present
    if pdb_id[:4] in db_pdbcodes:
        idx_to_remove = np.where(db_pdbcodes == pdb_id[:4])[0]
        db_pdbs = np.delete(db_pdbs, idx_to_remove)
        db_sources = np.delete(db_sources, idx_to_remove)
        db_labels = np.delete(db_labels, idx_to_remove)
        db_resids = np.delete(db_resids, idx_to_remove)
        db_embeddings = np.delete(db_embeddings, idx_to_remove, 0)
        db_means = np.delete(db_means, idx_to_remove, 0)
        db_stds = np.delete(db_stds, idx_to_remove, 0)
        db_cutoffs = np.delete(db_cutoffs, idx_to_remove, 0)
    
    # Extract data from the processed dictionary
    resids = np.array(processed_pdb_data['resids'])
    chains = np.array(processed_pdb_data['chains'])
    embeddings = np.array(processed_pdb_data['embeddings'])
    confidences = np.array(processed_pdb_data['confidence'])
    
    if af_flag:
        print('Removing low confidence residues')
        high_conf_idx = confidences >= 70
        resids = resids[high_conf_idx]
        chains = chains[high_conf_idx]
        embeddings = embeddings[high_conf_idx]
    
    if args.chains:
        print(f'Annotating chains: {args.chains}')
        chain_idx = np.in1d(chains, np.array(list(args.chains)))
        resids = resids[chain_idx]
        chains = chains[chain_idx]
        embeddings = embeddings[chain_idx]
        
    # Before cosine calculation
    print("\nInput statistics:")
    print(f"- Number of residues: {len(embeddings)}")
    print(f"- Embedding dimension: {embeddings.shape[1]}")
    print(f"- Sample embedding mean/std: {embeddings.mean():.6f}/{embeddings.std():.6f}")
    
    cosines = fastdist.cosine_matrix_to_matrix(embeddings, db_embeddings)  # (n_res, n_db)
    
    query_mask = cosines > cutoff
    site_mask = cosines > db_cutoffs[np.newaxis, :]
    
    quantile_mask = query_mask & site_mask

    results = col.defaultdict(dict)
    hit_idx_by_row = [np.nonzero(row)[0] for row in quantile_mask]
    
    for i, hit_idx in enumerate(hit_idx_by_row):
        if len(hit_idx) == 0:
            continue
        hits = np.unique(hit_idx)
        chain_res = chains[i] + '_' + resids[i]
        for h in hits:
            key = (db_labels[h], db_sources[h])
            if chain_res in results[key]:
                results[key][chain_res].add(f'{db_pdbs[h]}: {db_resids[h]}')
            else:
                results[key][chain_res] = set([f'{db_pdbs[h]}: {db_resids[h]}'])
    
    # After cosine calculation
    print("\nSimilarity statistics:")
    print(f"- Cosine matrix shape: {cosines.shape}")
    print(f"- Max similarity: {cosines.max():.6f}")
    print(f"- Mean similarity: {cosines.mean():.6f}")
    print(f"- Number of hits above cutoff: {query_mask.sum()}")
    print(f"- Number of hits above site cutoff: {site_mask.sum()}")
    print(f"- Number of final hits: {quantile_mask.sum()}")

    # After processing hits
    print("\nResults statistics:")
    print(f"- Number of result keys: {len(results)}")
    for (name, source), sites in results.items():
        print(f"- {name} ({source}): {sum(len(pdbs) for pdbs in sites.values())} total hits")
        for loc, pdbs in sites.items():
            if args.verbose:
                print(f"    - {loc}: {pdbs}")
            else:
                print(f"    - {loc}: {len(pdbs)} PDBs")
