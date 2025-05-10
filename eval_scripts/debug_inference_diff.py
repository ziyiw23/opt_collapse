# debug_inference_diff.py

import os
import sys
import io
import numpy as np
import pandas as pd
import torch
import random
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import argparse
import time
from contextlib import nullcontext, ExitStack
import collections as col
import glob
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import lmdb
import pickle
import gzip
import shutil
from tqdm import tqdm
import pprint

# python ./eval_scripts/debug_inference_diff.py --pdb_dir ../clps_pdbs --checkpoint ./data/checkpoints/collapse_base.pt --num_pdbs 20 --num_workers 9

# --- Seeding and Determinism ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA determinism enabled.")
# -----------------------------

# Ensure the main project directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Pipeline Imports ---
from collapse.data import (
    process_pdb as original_process_pdb,
    atom_info,
    BaseTransform as OriginalBaseTransform,
    extract_env_from_resid as original_extract_env,
    sample_functional_center # Shared function
)
from collapse import initialize_model
from embedding_utils import (
    GraphPreparationTransformCPU,
    TransformedDatasetWrapper,
    graph_collate_fn # The one returning 4 items
)
from atom3d.filters.filters import first_model_filter # Needed by original path

# --- Configuration ---
NUM_PDBS_TO_TEST = 20 # <<<--- Set to 20
NUM_WORKERS_OPTIMIZED = 9 # <<<--- Set to 9 workers for optimized graph gen
PREFETCH_FACTOR = 2
DEFAULT_PDB_DIR = "/scratch/groups/rbaltman/ziyiw23/1000_pdbs/" # Default, can override
CHECKPOINT_PATH = 'data/checkpoints/collapse_base.pt'
ENV_RADIUS = 10.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INCLUDE_HETS = False
ORIGINAL_EDGE_CUTOFF = 4.5
TEMP_LMDB_DIR = "temp_res" # Directory for temporary LMDBs

# --- Helper: Comparison Metrics ---
def calculate_embedding_comparison(pdb_id, stage, embs1_np, embs2_np):
    """Calculates MSE, Cosine Similarity, and L2 Norm Difference for average embeddings."""
    print(f"\n--- Comparing Average Embeddings for PDB: {pdb_id} (Stage: {stage}) ---")
    if embs1_np is None or embs2_np is None:
        print("  ❌ FAIL: One or both average embedding arrays are None.")
        return None
    # Ensure inputs are single vectors (1, dim)
    if embs1_np.ndim == 1: embs1_np = embs1_np.reshape(1, -1)
    if embs2_np.ndim == 1: embs2_np = embs2_np.reshape(1, -1)

    if embs1_np.shape != embs2_np.shape:
        print(f"  ❌ FAIL: Average embedding shapes differ! {embs1_np.shape} vs {embs2_np.shape}")
        return None

    print(f"  Average embedding shapes match: {embs1_np.shape}")
    embs1_np = embs1_np.astype(np.float32)
    embs2_np = embs2_np.astype(np.float32)

    mse = np.mean((embs1_np - embs2_np)**2)
    cos_sim = cosine_similarity(embs1_np, embs2_np)[0, 0] if not (np.isnan(embs1_np).any() or np.isnan(embs2_np).any()) else np.nan
    l2_diff = np.linalg.norm(embs1_np - embs2_np)

    print(f"  Mean Squared Error (MSE): {mse:.6g}")
    print(f"  Cosine Similarity       : {cos_sim:.6f}")
    print(f"  L2 Norm Difference    : {l2_diff:.6f}")
    return {'mse': mse, 'cosine_sim': cos_sim, 'l2_diff': l2_diff}

# --- Helper: LMDB Writing (atom3d style) ---
def write_lmdb(output_dir, data_dict, map_size=int(1e11)): # Smaller map size for debug
    """Writes processed data to LMDB in atom3d format."""
    print(f"\nWriting {len(data_dict)} entries to LMDB: {output_dir}")
    if not data_dict: return False
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    serialization_format = 'pkl'
    try:
        env = lmdb.open(output_dir, map_size=map_size)
        id_to_idx = {}
        final_count = 0
        with env.begin(write=True) as txn:
            for i, (protein_id, item_data) in enumerate(tqdm(data_dict.items(), desc="Writing LMDB")):
                # item_data should be {'id':.., 'embeddings':.., 'resids':.., etc.}
                item_to_save = item_data.copy()
                item_to_save['types'] = {key: str(type(val)) for key, val in item_to_save.items()}
                item_to_save['types']['types'] = str(type(item_to_save['types']))

                try:
                    serialized_data = pickle.dumps(item_to_save, protocol=pickle.HIGHEST_PROTOCOL)
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f: f.write(serialized_data)
                    compressed_value = buf.getvalue()
                    key = str(i).encode('utf-8')
                    put_success = txn.put(key, compressed_value, overwrite=False)
                    if not put_success: raise RuntimeError(f'LMDB entry {i} exists')
                    id_to_idx[protein_id] = i
                    final_count += 1
                except Exception as write_e:
                    print(f"\nError writing {protein_id} (index {i}): {write_e}")
                    continue

            print(f"\nWriting LMDB metadata for {final_count} items...")
            txn.put(b'num_examples', str(final_count).encode())
            txn.put(b'serialization_format', serialization_format.encode())
            id_to_idx_serialized = pickle.dumps(id_to_idx, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(b'id_to_idx', id_to_idx_serialized)
        env.close()
        print(f"LMDB writing complete. Wrote {final_count} items.")
        return True
    except Exception as e:
        print(f"\nError during LMDB write transaction: {e}")
        traceback.print_exc()
        return False

# --- Helper: LMDB Reading ---
def read_lmdb_embeddings(lmdb_dir):
    """Reads embeddings from LMDB, returns dict {protein_id: avg_embedding}."""
    print(f"\nReading average embeddings from LMDB: {lmdb_dir}")
    embeddings_map = {}
    try:
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            id_to_idx_serialized = txn.get(b'id_to_idx')
            num_examples_bytes = txn.get(b'num_examples')
            if id_to_idx_serialized is None or num_examples_bytes is None:
                print("  Error: Metadata (id_to_idx or num_examples) not found.")
                env.close()
                return {}
            id_to_idx = pickle.loads(id_to_idx_serialized)
            num_examples = int(num_examples_bytes.decode())

            for protein_id, idx in tqdm(id_to_idx.items(), desc="Reading LMDB"):
                key = str(idx).encode('utf-8')
                value = txn.get(key)
                if value:
                    try:
                        data = pickle.loads(gzip.decompress(value))
                        if 'embeddings' in data and data['embeddings'] is not None:
                            emb = data['embeddings']
                            if isinstance(emb, np.ndarray):
                                # Average if 2D, use directly if 1D
                                if emb.ndim == 2 and emb.shape[0] > 0:
                                    avg_emb = emb.astype(np.float32).mean(axis=0)
                                elif emb.ndim == 1 and emb.shape[0] > 0:
                                    avg_emb = emb.astype(np.float32)
                                else: continue # Skip empty or weird shapes
                                embeddings_map[protein_id] = avg_emb
                    except Exception as read_e:
                        print(f"\nError reading/processing entry for {protein_id} (key {idx}): {read_e}")
        env.close()
    except Exception as e:
        print(f"\nError opening/reading LMDB {lmdb_dir}: {e}")
        traceback.print_exc()
    print(f"Read {len(embeddings_map)} average embeddings.")
    return embeddings_map

# --- Helper: Inference Function ---
def run_inference(model, graphs_list, device):
    """Runs inference on a list of graphs for a single protein."""
    if not graphs_list: return None
    try:
        batch = Batch.from_data_list(graphs_list).to(device)
        with torch.no_grad():
             # Use the correct encoder call, returning only embeddings
             embs_tensor, _ = model.online_encoder(batch, return_projection=False) # CORRECTED CALL
             embs_np = embs_tensor.float().cpu().numpy()
        del batch, embs_tensor
        if device.type == 'cuda': torch.cuda.empty_cache()
        return embs_np
    except AttributeError as ae:
         # Add specific handling for this expected error if the fix doesn't work everywhere
         print(f"  AttributeError during inference: {ae}. Model type: {type(model)}")
         if device.type == 'cuda': torch.cuda.empty_cache()
         return None
    except Exception as e:
        print(f"  Inference Error: {e}")
        if device.type == 'cuda': torch.cuda.empty_cache()
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Original vs Optimized Embedding Pipeline")
    parser.add_argument('--pdb_dir', type=str, default=DEFAULT_PDB_DIR, help="Directory containing PDB files")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    parser.add_argument('--num_pdbs', type=int, default=NUM_PDBS_TO_TEST, help="Number of PDBs to test")
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS_OPTIMIZED, help="Number of workers for optimized graph gen")
    parser.add_argument('--output_dir', type=str, default=TEMP_LMDB_DIR, help="Base directory for temporary LMDBs")
    parser.add_argument('--no_compile', action='store_true', help="Disable torch.compile for optimized path")
    args = parser.parse_args()

    print(f"--- Configuration ---")
    print(f"PDB Directory: {args.pdb_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of PDBs: {args.num_pdbs}")
    print(f"Optimized Workers: {args.num_workers}")
    print(f"Temp Output Dir: {args.output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Compile Optimized Model: {not args.no_compile}")
    print("-" * 20)

    # --- Phase 0: Setup ---
    # Find PDB files
    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, '*.pdb*')))
    if not pdb_files:
        print(f"Error: No PDB files found in {args.pdb_dir}")
        sys.exit(1)
    pdb_files_to_process = pdb_files[:args.num_pdbs]
    pdb_ids_to_process = [os.path.basename(f).split('.')[0] for f in pdb_files_to_process]
    print(f"Processing PDB IDs: {pdb_ids_to_process}")

    # Load Model
    print("\nLoading model...")
    model = initialize_model(args.checkpoint, device=DEVICE).eval()
    print("Model loaded.")

    # Compile model if requested
    model_for_opt_inference = model
    if not args.no_compile and hasattr(torch, 'compile'):
        print("Compiling model for optimized path inference...")
        try:
            model_for_opt_inference = torch.compile(model, mode="default")
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}. Using uncompiled model.")
    elif args.no_compile:
         print("Skipping model compilation as requested.")

    # Prepare output directories
    lmdb_ori_path = os.path.join(args.output_dir, "ori")
    lmdb_opt_path = os.path.join(args.output_dir, "opt")
    if os.path.exists(lmdb_ori_path): shutil.rmtree(lmdb_ori_path)
    if os.path.exists(lmdb_opt_path): shutil.rmtree(lmdb_opt_path)
    os.makedirs(lmdb_ori_path, exist_ok=True)
    os.makedirs(lmdb_opt_path, exist_ok=True)

    # Data storage
    original_graphs = {} # {pdb_id: [graphs]}
    original_metadata = {} # {pdb_id: [metadata_dicts]}
    optimized_graphs = {}
    optimized_metadata = {}
    original_atoms_map = {} # Keep original atoms for LMDB writing
    optimized_atoms_map = {}

    # --- Phase 1: Graph Generation ---
    print(f"\n{'='*10} Phase 1: Graph Generation {'='*10}")

    # --- 1a: Original Graph Generation (Sequential) ---
    print("\nGenerating graphs using ORIGINAL method...")
    original_base_transform = OriginalBaseTransform(edge_cutoff=ORIGINAL_EDGE_CUTOFF, device='cpu') # Use original BaseTransform
    start_time_orig_graph = time.time()
    for pdb_file in tqdm(pdb_files_to_process, desc="Original Graphs"):
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        graphs = []
        metadata_list = []
        try:
            atom_df = original_process_pdb(pdb_file, include_hets=INCLUDE_HETS) # Use original processing
            if atom_df is None or atom_df.empty: continue
            original_atoms_map[pdb_id] = atom_df # Save for LMDB
            # Iterate through residues like original embed_protein
            for (c, i, r), res_df in atom_df.groupby(['chain', 'residue', 'resname']):
                if r not in atom_info.aa[:20]: continue # Skip non-standard
                resid = atom_info.aa_to_letter(r) + str(i)
                chain_atoms = atom_df[atom_df.chain == c] # Get atoms for the specific chain
                # Use original extract_env_from_resid which uses the original BaseTransform
                out = original_extract_env(chain_atoms, (c, resid), env_radius=ENV_RADIUS, res_df=res_df.copy())
                if out is not None and out[0] is not None:
                    graph, _ = out
                    graphs.append(graph)
                    metadata_list.append({
                        'protein_id': pdb_id, 'chain': c, 'resid': resid,
                        'confidence': float(res_df['bfactor'].iloc[0]) if 'bfactor' in res_df.columns else 0.0
                    })
            original_graphs[pdb_id] = graphs
            original_metadata[pdb_id] = metadata_list
        except Exception as e:
            print(f"Error processing {pdb_id} (Original): {e}")
            # traceback.print_exc()
    end_time_orig_graph = time.time()
    print(f"Original graph generation took: {end_time_orig_graph - start_time_orig_graph:.2f}s")

    # --- 1b: Optimized Graph Generation (Parallel) ---
    print("\nGenerating graphs using OPTIMIZED method (parallel)...")
    # Need a dataset source for the DataLoader
    class SimplePDBDataset(Dataset):
        def __init__(self, file_list): self.file_list = file_list
        def __len__(self): return len(self.file_list)
        def __getitem__(self, idx):
            filepath = self.file_list[idx]
            pdb_id = os.path.basename(filepath).split('.')[0]
            try:
                # Use the *original* PDB processing here too for consistency before transform
                atoms = original_process_pdb(filepath, include_hets=INCLUDE_HETS)
                if atoms is None: return None # Handle case where process_pdb fails
                return {'id': pdb_id, 'atoms': atoms, 'file_path': filepath}
            except Exception as e:
                print(f"Error loading raw atoms for {pdb_id} in SimplePDBDataset: {e}")
                return None

    opt_dataset_raw = SimplePDBDataset(pdb_files_to_process)
    opt_transform_cpu = GraphPreparationTransformCPU(
         include_hets=INCLUDE_HETS, env_radius=ENV_RADIUS # Use 32 for consistency if needed
    )
    opt_dataset_transformed = TransformedDatasetWrapper(opt_dataset_raw, opt_transform_cpu)
    opt_dataloader = DataLoader(
        opt_dataset_transformed, batch_size=1, # Process one PDB per worker task
        shuffle=False, num_workers=args.num_workers,
        collate_fn=graph_collate_fn, # Returns 4 items
        pin_memory=False, 
        prefetch_factor=PREFETCH_FACTOR if args.num_workers > 0 else 2
    )

    start_time_opt_graph = time.time()
    for batch_data in tqdm(opt_dataloader, desc="Optimized Graphs"):
        if batch_data is None: continue
        graph_batch, metadata_list, atoms_map_batch, filepath_map_batch = batch_data
        if graph_batch is None or not metadata_list: continue

        # Since batch_size=1 for dataloader, maps should have one entry
        pdb_id = metadata_list[0]['protein_id']
        optimized_graphs[pdb_id] = graph_batch.to_data_list() # Store list of graphs
        optimized_metadata[pdb_id] = metadata_list
        if pdb_id in atoms_map_batch: optimized_atoms_map[pdb_id] = atoms_map_batch[pdb_id]

    end_time_opt_graph = time.time()
    print(f"Optimized graph generation took: {end_time_opt_graph - start_time_opt_graph:.2f}s")

    # --- 1c: Graph Comparison ---
    print("\nComparing Graph Generation Results...")
    graph_comparison_summary = {'match': 0, 'mismatch': 0, 'missing_ori': 0, 'missing_opt': 0}
    for pdb_id in pdb_ids_to_process:
        print(f"  Comparing {pdb_id}:")
        ori_g = original_graphs.get(pdb_id)
        opt_g = optimized_graphs.get(pdb_id)
        ori_m = original_metadata.get(pdb_id)
        opt_m = optimized_metadata.get(pdb_id)

        if ori_g is None and opt_g is None:
            print("    - Both methods failed to generate graphs.")
            continue
        elif ori_g is None:
            print("    - ❌ FAIL: Original failed, Optimized succeeded.")
            graph_comparison_summary['missing_ori'] += 1
            continue
        elif opt_g is None:
            print("    - ❌ FAIL: Optimized failed, Original succeeded.")
            graph_comparison_summary['missing_opt'] += 1
            continue

        # Compare counts
        print(f"    - Original Graph Count: {len(ori_g)}")
        print(f"    - Optimized Graph Count: {len(opt_g)}")
        count_mismatch = len(ori_g) != len(opt_g)

        # Compare residue IDs
        ori_resids = sorted([m['resid'] for m in ori_m])
        opt_resids = sorted([m['resid'] for m in opt_m])
        resid_mismatch = ori_resids != opt_resids

        if count_mismatch or resid_mismatch:
            print(f"    - ❌ FAIL: Mismatch found.")
            if count_mismatch: print(f"      - Graph counts differ.")
            if resid_mismatch:
                 print(f"      - Residue ID lists differ.")
                 # Optionally print details
                 # print(f"        Original : {ori_resids}")
                 # print(f"        Optimized: {opt_resids}")
            graph_comparison_summary['mismatch'] += 1
        else:
            print(f"    - ✅ MATCH: Graph counts and residue IDs match.")
            graph_comparison_summary['match'] += 1

    print(f"Graph Comparison Summary: {graph_comparison_summary}")


    # --- Phase 2: Direct Inference & Comparison ---
    print(f"\n{'='*10} Phase 2: Direct Inference {'='*10}")
    original_embeddings_direct = {} # {pdb_id: avg_embedding_np}
    optimized_embeddings_direct = {}

    # --- 2a: Original Inference ---
    print("\nRunning inference on ORIGINAL graphs...")
    start_time_orig_inf = time.time()
    for pdb_id in tqdm(original_graphs, desc="Original Inference"):
        embs = run_inference(model, original_graphs[pdb_id], DEVICE)
        if embs is not None and embs.ndim == 2 and embs.shape[0] > 0:
            original_embeddings_direct[pdb_id] = embs.mean(axis=0)
        else: print(f" Failed inference for {pdb_id}")
    end_time_orig_inf = time.time()
    print(f"Original inference took: {end_time_orig_inf - start_time_orig_inf:.2f}s")

    # --- 2b: Optimized Inference ---
    print("\nRunning inference on OPTIMIZED graphs...")
    start_time_opt_inf = time.time()
    # Use the potentially compiled model
    with ExitStack() as stack:
        # Apply autocast only if on CUDA and compiling didn't fail
        autocast_enabled = (DEVICE.type == 'cuda' and model_for_opt_inference is not model)
        autocast_context = torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=autocast_enabled)
        stack.enter_context(autocast_context)

        for pdb_id in tqdm(optimized_graphs, desc="Optimized Inference"):
             embs = run_inference(model_for_opt_inference, optimized_graphs[pdb_id], DEVICE)
             if embs is not None and embs.ndim == 2 and embs.shape[0] > 0:
                 optimized_embeddings_direct[pdb_id] = embs.mean(axis=0)
             else: print(f" Failed inference for {pdb_id}")
    end_time_opt_inf = time.time()
    print(f"Optimized inference took: {end_time_opt_inf - start_time_opt_inf:.2f}s")


    # --- 2c: Direct Embedding Comparison ---
    print("\nComparing Direct Inference Results (Average Embeddings)...")
    direct_comparison_results = {}
    for pdb_id in pdb_ids_to_process:
        if pdb_id in original_embeddings_direct and pdb_id in optimized_embeddings_direct:
            comparison = calculate_embedding_comparison(
                pdb_id, "Direct Inference",
                original_embeddings_direct[pdb_id],
                optimized_embeddings_direct[pdb_id]
            )
            if comparison: direct_comparison_results[pdb_id] = comparison
        else:
            print(f"  Skipping comparison for {pdb_id}: Missing result from one or both paths.")

    # --- Phase 3: LMDB Generation & Comparison ---
    print(f"\n{'='*10} Phase 3: LMDB Storage & Retrieval {'='*10}")

    # --- 3a: Prepare Data for LMDB ---
    # We need per-residue embeddings and metadata to write LMDB
    # Rerun inference to get per-residue embeddings (or store them from Phase 2)
    # Let's rerun for simplicity here, could optimize by storing earlier
    original_data_for_lmdb = {}
    print("\nRe-running inference for ORIGINAL per-residue embeddings...")
    for pdb_id in tqdm(original_graphs, desc="Original LMDB Data Prep"):
        embs = run_inference(model, original_graphs[pdb_id], DEVICE)
        if embs is not None and len(original_metadata[pdb_id]) == embs.shape[0]:
             original_data_for_lmdb[pdb_id] = {
                 'id': pdb_id,
                 'embeddings': embs,
                 'resids': [m['resid'] for m in original_metadata[pdb_id]],
                 'chains': [m['chain'] for m in original_metadata[pdb_id]],
                 'confidence': [m['confidence'] for m in original_metadata[pdb_id]],
                 'atoms': original_atoms_map.get(pdb_id), # Add atoms
                 'file_path': None # Original doesn't store path explicitly here
             }

    optimized_data_for_lmdb = {}
    print("\nRe-running inference for OPTIMIZED per-residue embeddings...")
    with ExitStack() as stack:
        autocast_enabled = (DEVICE.type == 'cuda' and model_for_opt_inference is not model)
        autocast_context = torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=autocast_enabled)
        stack.enter_context(autocast_context)
        for pdb_id in tqdm(optimized_graphs, desc="Optimized LMDB Data Prep"):
            embs = run_inference(model_for_opt_inference, optimized_graphs[pdb_id], DEVICE)
            if embs is not None and len(optimized_metadata[pdb_id]) == embs.shape[0]:
                 optimized_data_for_lmdb[pdb_id] = {
                     'id': pdb_id,
                     'embeddings': embs,
                     'resids': [m['resid'] for m in optimized_metadata[pdb_id]],
                     'chains': [m['chain'] for m in optimized_metadata[pdb_id]],
                     'confidence': [m['confidence'] for m in optimized_metadata[pdb_id]],
                     'atoms': optimized_atoms_map.get(pdb_id), # Add atoms
                     'file_path': None # File path not easily available here post-dataloader
                 }

    # --- 3b: Write LMDBs ---
    write_lmdb(lmdb_ori_path, original_data_for_lmdb)
    write_lmdb(lmdb_opt_path, optimized_data_for_lmdb)

    # --- 3c: Read LMDBs ---
    original_embeddings_lmdb = read_lmdb_embeddings(lmdb_ori_path)
    optimized_embeddings_lmdb = read_lmdb_embeddings(lmdb_opt_path)

    # --- 3d: LMDB Embedding Comparison ---
    print("\nComparing LMDB Retrieval Results (Average Embeddings)...")
    lmdb_comparison_results = {}
    common_lmdb_keys = sorted(list(set(original_embeddings_lmdb.keys()) & set(optimized_embeddings_lmdb.keys())))
    print(f"Found {len(common_lmdb_keys)} common protein IDs in LMDB results.")

    for pdb_id in common_lmdb_keys:
        comparison = calculate_embedding_comparison(
            pdb_id, "LMDB Retrieval",
            original_embeddings_lmdb[pdb_id],
            optimized_embeddings_lmdb[pdb_id]
        )
        if comparison: lmdb_comparison_results[pdb_id] = comparison

    # --- Phase 4: Cleanup ---
    print(f"\n{'='*10} Phase 4: Cleanup {'='*10}")
    try:
        print(f"Removing temporary directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
        print("Cleanup successful.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

    print(f"\n{'='*20} DEBUGGING COMPLETE {'='*20}")
