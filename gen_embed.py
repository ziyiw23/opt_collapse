# V2.5 - Refactored for Pickling & Modified for Original Output Format

import numpy as np
import os
import argparse
import torch
import torch.nn as nn # Keep for DataParallel
import pandas as pd
from atom3d.datasets import load_dataset # Keep this
import atom3d.util.file as fi
from collapse import initialize_model, atom_info # Assuming these are correct
# from atom3d.filters.filters import first_model_filter # Moved to utils
import collections as col
import random
# import torch_cluster # Only needed if BaseTransformCPU uses it
from torch_geometric.data import Batch, Data # Batch needed for collate
from torch.utils.data import Dataset, DataLoader # Keep DataLoader
import lmdb # For manual LMDB writing
import pickle # For LMDB serialization
from tqdm import tqdm # Progress bar
import gzip # For compression
import io
# from scipy.spatial import KDTree # Moved to utils

import time
import sys

# --- DDP Imports --- ## REMOVED ##
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# --- Import from utils --- ## MODIFIED ##
from embedding_utils import (
    BaseTransform, GraphPreparationTransformCPU, TransformedDatasetWrapper,
    graph_collate_fn # Now returns 4 items
)

# --- DDP Helper Functions --- ## REMOVED ##
# def setup_ddp():
#     """Initializes the distributed process group."""
#     # Assumes environment variables (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are set by launcher
#     if not dist.is_available():
#         raise RuntimeError("Distributed training requires torch.distributed.")
#     if not dist.is_initialized():
#         # Default backend for multi-GPU node is NCCL
#         dist.init_process_group(backend='nccl', init_method='env://')

# def cleanup_ddp():
#     """Cleans up the distributed process group."""
#     if dist.is_initialized():
#         dist.destroy_process_group()

# def is_main_process():
#     """Checks if the current process is the main one (rank 0)."""
#     if not dist.is_initialized(): return True # Not distributed
#     return dist.get_rank() == 0

# --- Seeding and Constants ---
seed = 42
# Seed needs to be set consistently across processes if randomness matters early
# random.seed(seed) # Might need adjustments for DDP if used before sampler
# np.random.seed(seed) # Might need adjustments for DDP
# torch.manual_seed(seed) # Set per process? Generally okay.
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed) # Sets for all GPUs, potentially redundant with set_device?
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False # Deterministic takes priority
# --- Seeding needs careful review for DDP if perfect reproducibility is critical ---
# --- For now, assume default seeding is sufficient for inference ---

# ELEMENT_MAPPING, DEFAULT_ELEMENT Moved to embedding_utils.py
# --- Helper Functions (_normalize, _rbf, _edge_features) Moved to embedding_utils.py ---
# --- BaseTransform (CPU Version for Workers) Moved to embedding_utils.py ---
# --- sample_functional_center Moved to embedding_utils.py ---
# --- extract_env_for_residue (CPU version using SciPy KDTree) Moved to embedding_utils.py ---
# --- prepare_graphs_for_protein (Worker Task Helper - CPU) Moved to embedding_utils.py ---
# --- Graph Preparation Transform (CPU Version for Workers) Moved to embedding_utils.py ---
# --- Custom Collate Function (Robust Version) ---
# Moved to embedding_utils.py
# --- Dataset Wrapper (Applies CPU Transform in Worker) Moved to embedding_utils.py ---


# --- is_valid_pdb ---
# (Keep as before)
def is_valid_pdb(filepath):
    try: return os.path.getsize(filepath) > 0
    except OSError: return False

# --- main function (V2.5 - Reverted to DataParallel) --- ## REVERTED ##
def main():
    parser = argparse.ArgumentParser(description="V2.5 Embedding generation - Saving Original Format - DataParallel") # Updated desc
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
    parser.add_argument('--num_workers', type=int, default=6, help="Number of DataLoader workers") # Reverted meaning
    parser.add_argument('--batch_size', type=int, default=1, help="Number of *proteins* per GPU batch (Recommend 1)")
    parser.add_argument('--prefetch_factor', type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # --- Device Setup (Simpler for DataParallel) --- ## REVERTED ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using primary device: {device}")
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices.")
    # --- End Device Setup ---

    # --- Seeding --- (Already handled above) ---

    # --- Print Initial Info ---
    print(f"Using num_workers: {args.num_workers}, batch_size: {args.batch_size}")
    print(f"Number of cpus available (OS level): {os.cpu_count()}")
    if args.batch_size > 1:
        print("WARNING: batch_size > 1 with original atom data saving can lead to high memory usage and slow collation.")

    # --- Load Model --- ## REVERTED ##
    print("Loading model...")
    model = initialize_model(args.checkpoint, device=device) # Load directly to primary device
    model.eval()

    # --- Wrap model for DataParallel --- ## RE-ADDED ##
    if torch.cuda.device_count() > 1:
        print(f"Wrapping model with nn.DataParallel for {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    else:
        print("Running on single device (CPU or single GPU). No DataParallel wrap needed.")
    # --- End DataParallel Wrap ---

    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        try:
            model = torch.compile(model, mode="default") # Compile after wrapping
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")

    # --- Load RAW Dataset Structure --- ## REVERTED ##
    print(f"Loading RAW dataset structure from: {args.data_dir} with filetype: {args.filetype}")
    try:
        raw_dataset_base = load_dataset(args.data_dir, args.filetype, transform=None)
        dataset_len = len(raw_dataset_base)
        print(f"Initial raw dataset size: {dataset_len}")
        if dataset_len == 0:
            print("Error: Loaded raw dataset is empty.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading raw dataset structure: {e}")
        sys.exit(1)

    # --- Instantiate the CPU Transform --- (Unchanged)
    graph_transform_cpu = GraphPreparationTransformCPU(
        include_hets=args.include_hets,
        env_radius=args.env_radius
    )

    # --- Wrap Dataset with Transform --- (Unchanged)
    dataset_for_loader = TransformedDatasetWrapper(raw_dataset_base, graph_transform_cpu)

    # --- Splitting Logic (Original) --- ## REVERTED ##
    indices = np.arange(len(dataset_for_loader))
    if args.num_splits > 1:
        if args.split_id < 1 or args.split_id > args.num_splits:
             print(f"Error: split_id ({args.split_id}) must be between 1 and {args.num_splits}")
             sys.exit(1)
        split_indices = np.array_split(indices, args.num_splits)[args.split_id - 1]
        if len(split_indices) == 0: print(f"Warning: Split {args.split_id} has 0 examples.")
        print(f'Processing split {args.split_id}/{args.num_splits} with {len(split_indices)} examples...')
        final_dataset_to_load = torch.utils.data.Subset(dataset_for_loader, split_indices)
    else:
        print(f'Processing full dataset with {len(dataset_for_loader)} examples...')
        final_dataset_to_load = dataset_for_loader
    # --- End Splitting ---

    out_path = args.out_dir
    if args.num_splits > 1:
         out_path = os.path.join(args.out_dir, f'embeddings_split_{args.split_id}')

    os.makedirs(out_path, exist_ok=True)

    # --- Setup DataLoader (No Sampler) --- ## REVERTED ##
    pin_memory_enabled = (str(device) != 'cpu') and (args.batch_size == 1)
    print(f"Pin memory enabled: {pin_memory_enabled}")

    dataloader = DataLoader(
        final_dataset_to_load,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_collate_fn,
        pin_memory=pin_memory_enabled,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,
        # sampler=sampler # Removed sampler
    )

    # --- Main Processing & Inference Loop --- ## REVERTED ##
    print("Starting DataParallel processing and LMDB writing...")
    start_loop_time = time.time()
    results_to_save = [] # Single list for results
    processed_protein_count = 0
    failed_protein_count = 0
    processed_residue_count = 0

    batch_iterator = tqdm(dataloader, desc="Processing Batches", total=len(dataloader))

    for batch_idx, batch_data in enumerate(batch_iterator):

        # --- Unpack batch data --- (Unchanged)
        if batch_data is None: continue
        graph_batch_cpu, metadata_batch, atoms_map, filepath_map = batch_data
        if graph_batch_cpu is None or metadata_batch is None or atoms_map is None or filepath_map is None: continue

        proteins_in_batch_ids = list(set(m['protein_id'] for m in metadata_batch))
        if not proteins_in_batch_ids: continue

        graphs_on_cpu_list = graph_batch_cpu.to_data_list()
        num_graphs_in_batch = len(graphs_on_cpu_list)

        # --- Attempt Normal Batch Inference ---
        final_embs_np = None
        batch_failed_oom = False
        try:
            if num_graphs_in_batch == 0:
                failed_protein_count += len(proteins_in_batch_ids)
                continue

            # Move graph batch to the primary device (DataParallel handles distribution)
            graph_batch_gpu = graph_batch_cpu.to(device)

            with torch.no_grad():
                # DataParallel handles the forward pass distribution
                with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                    # Call model directly (DP wrapper handles it)
                    # Or access original methods via model.module if needed, but usually direct call works
                    embs = model.online_network(graph_batch_gpu)
                    # Ensure embeddings are float32 for consistent saving
                    final_embs_np = embs.float().cpu().numpy()
                processed_residue_count += final_embs_np.shape[0]

            # Clear GPU memory quickly after successful inference
            del graph_batch_gpu
            del embs
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                batch_failed_oom = True # Set flag to trigger fallback
                print(f"\nOOM on batch {batch_idx}. Batch size: {args.batch_size}, Num graphs: {num_graphs_in_batch}. Triggering fallback...")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                final_embs_np = None
            else:
                print(f"\nRuntime error on Batch {batch_idx}: {e}. Failing proteins: {proteins_in_batch_ids}")
                failed_protein_count += len(proteins_in_batch_ids)
                continue
        except Exception as e:
            print(f"\nNon-runtime error on Batch {batch_idx}: {e}. Failing proteins: {proteins_in_batch_ids}")
            failed_protein_count += len(proteins_in_batch_ids)
            continue

        # --- Process Results (Normal or Fallback) ---
        results_this_batch = col.defaultdict(lambda: col.defaultdict(list))

        if not batch_failed_oom:
            # --- Normal Processing --- (Unchanged logic)
            if final_embs_np is None:
                 print(f"\nWarning: Embeddings None (Batch {batch_idx}). Skipping.")
                 failed_protein_count += len(proteins_in_batch_ids)
                 continue
            if len(metadata_batch) != final_embs_np.shape[0]:
                 print(f"\nCRITICAL WARNING: Metadata/Emb len mismatch ({len(metadata_batch)} vs {final_embs_np.shape[0]}) Batch {batch_idx}. Skipping.")
                 failed_protein_count += len(proteins_in_batch_ids)
                 continue

            for i, meta in enumerate(metadata_batch):
                protein_id = meta['protein_id']
                results_this_batch[protein_id]['resids'].append(meta['resid'])
                results_this_batch[protein_id]['chains'].append(meta['chain'])
                results_this_batch[protein_id]['confidence'].append(meta['confidence'])
                results_this_batch[protein_id]['embeddings'].append(final_embs_np[i])

        else:
            # --- OOM Fallback Processing --- ## REVERTED (use model.module) ##
            print(f"--- Starting OOM Fallback for Batch {batch_idx} ---")
            graphs_by_protein = col.defaultdict(list)
            metadata_by_protein = col.defaultdict(list)
            if len(graphs_on_cpu_list) != len(metadata_batch):
                 print(f"  ❌ CRITICAL OOM Fallback Mismatch. Skipping batch {batch_idx}.")
                 failed_protein_count += len(proteins_in_batch_ids)
                 continue

            for graph, meta in zip(graphs_on_cpu_list, metadata_batch):
                graphs_by_protein[meta['protein_id']].append(graph)
                metadata_by_protein[meta['protein_id']].append(meta)

            for protein_id in proteins_in_batch_ids:
                 # ... (OOM fallback logic, ensuring calls are like model.module.online_encoder) ...
                if protein_id not in graphs_by_protein or protein_id not in metadata_by_protein:
                    print(f"  Warning: Protein {protein_id} missing during OOM fallback. Skipping.")
                    failed_protein_count += 1
                    continue

                protein_graphs = graphs_by_protein[protein_id]
                protein_metadata = metadata_by_protein[protein_id]
                num_residues = len(protein_graphs)
                if num_residues == 0: continue

                chunk_size = 128 # Keep user's change
                protein_embeddings_list = []
                protein_failed_chunking = False

                for i in range(0, num_residues, chunk_size):
                    chunk_graphs = protein_graphs[i : i + chunk_size]
                    if not chunk_graphs: continue

                    try:
                        # Move chunk to primary GPU for fallback
                        chunk_batch = Batch.from_data_list(chunk_graphs).to(device)
                        with torch.no_grad():
                            with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                                # Use model.module with DataParallel wrapper
                                # Check if model is wrapped before accessing .module
                                if isinstance(model, nn.DataParallel):
                                    chunk_embs, _ = model.module.online_encoder(chunk_batch, return_projection=False)
                                else:
                                    chunk_embs, _ = model.online_encoder(chunk_batch, return_projection=False)
                                protein_embeddings_list.append(chunk_embs.float().cpu().numpy())
                        del chunk_batch
                        del chunk_embs
                        if torch.cuda.is_available(): torch.cuda.empty_cache() # Need aggressive clearing here?

                    except RuntimeError as chunk_e:
                        if "CUDA out of memory" in str(chunk_e):
                            print(f"  ❌ OOM during chunking fallback for {protein_id} (chunk {i}). Protein failed.")
                        else:
                            print(f"  ❌ Runtime error during fallback chunk for {protein_id}: {chunk_e}")
                        protein_failed_chunking = True
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        break
                    except Exception as chunk_e:
                        print(f"  ❌ Non-runtime error during fallback chunk for {protein_id}: {chunk_e}")
                        protein_failed_chunking = True
                        break

                if not protein_failed_chunking and protein_embeddings_list:
                    try:
                        all_protein_embs = np.concatenate(protein_embeddings_list, axis=0)
                        if all_protein_embs.shape[0] == num_residues:
                            mean_embedding = np.mean(all_protein_embs, axis=0, dtype=np.float32)
                            results_this_batch[protein_id]['embeddings'] = [mean_embedding]
                            results_this_batch[protein_id]['pooling_type'] = ['mean_oom_fallback']
                            results_this_batch[protein_id]['resids'] = [protein_metadata[0]['resid']]
                            results_this_batch[protein_id]['chains'] = [protein_metadata[0]['chain']]
                            results_this_batch[protein_id]['confidence'] = [np.mean([m['confidence'] for m in protein_metadata])]
                            print(f"  ✓ Fallback successful for {protein_id} (Mean Pooled).")
                        else:
                            print(f"  ❌ Mismatch after fallback concatenation for {protein_id}. Skipping.")
                            protein_failed_chunking = True
                    except Exception as pool_e:
                        print(f"  ❌ Error during pooling/storing fallback for {protein_id}: {pool_e}")
                        protein_failed_chunking = True

                if protein_failed_chunking:
                    failed_protein_count += 1

            print(f"--- Finished OOM Fallback for Batch {batch_idx} ---")
            if torch.cuda.is_available(): torch.cuda.empty_cache()


        # --- Finalize and Add Results from Batch to results_to_save --- ## REVERTED ##
        for protein_id, data in results_this_batch.items():
            if not data.get('embeddings'): continue
            final_embedding_data = None
            # ... (Stacking/Handling logic as before) ...
            if isinstance(data['embeddings'], list):
                if 'pooling_type' not in data:
                    try:
                        if data['embeddings']:
                            final_embedding_data = np.stack(data['embeddings'], axis=0)
                            if len(data.get('resids', [])) != final_embedding_data.shape[0]:
                                print(f"\nFinal internal mismatch for {protein_id} (normal). Skipping.")
                                failed_protein_count += 1
                                continue
                        else:
                             print(f"\nWarning: Empty embedding list for {protein_id} (normal). Skipping.")
                             failed_protein_count += 1
                             continue
                    except Exception as stack_e:
                        print(f"\nError stacking embeddings for {protein_id} (normal): {stack_e}")
                        failed_protein_count += 1
                        continue
                else:
                    if data['embeddings']:
                        final_embedding_data = data['embeddings'][0]
                    else:
                         print(f"\nWarning: Empty embedding list for {protein_id} (fallback). Skipping.")
                         failed_protein_count += 1
                         continue
            else:
                 print(f"\nWarning: Unexpected embedding type for {protein_id}. Skipping.")
                 failed_protein_count += 1
                 continue

            if final_embedding_data is not None:
                final_protein_dict = {'id': protein_id}
                final_protein_dict.update(data)
                final_protein_dict['embeddings'] = final_embedding_data

                if protein_id in atoms_map and atoms_map[protein_id] is not None:
                    final_protein_dict['atoms'] = atoms_map[protein_id]

                if protein_id in filepath_map and filepath_map[protein_id] is not None:
                    final_protein_dict['file_path'] = filepath_map[protein_id]

                results_to_save.append(final_protein_dict) # Add to single list
                processed_protein_count += 1

    # --- End Main Loop ---

    # --- No Aggregation Needed for DataParallel --- ## REMOVED ##

    loop_end_time = time.time()
    print("-" * 30)
    print(f"Embedding generation loop finished in {loop_end_time - start_loop_time:.2f} seconds.")
    print(f"Total successfully processed proteins: {processed_protein_count}")
    print(f"Total residues processed (embeddings generated): {processed_residue_count}")
    if failed_protein_count > 0:
        print(f"Total Failed/Skipped proteins (OOM or other error): {failed_protein_count}")
    print("-" * 30)

    # --- Write results to LMDB (Single Process) --- ## REVERTED ##
    print(f"Writing {len(results_to_save)} results to LMDB directory: {out_path}")
    if not results_to_save:
        print("No results to save.")
    else:
        map_size = int(1024 * 1024 * 1024 * 50) # 50 GB
        serialization_format = 'pkl'
        try:
            # Pass the directory path (out_path) to lmdb.open, not the specific file path
            env = lmdb.open(out_path, map_size=map_size, subdir=True, writemap=True, meminit=False, map_async=True)
            with env.begin(write=True) as txn:
                lmdb_count = 0
                id_to_idx = {}
                try:
                    for i, result_dict in enumerate(tqdm(results_to_save, desc="Writing LMDB")):
                        item_to_save = result_dict
                        try:
                            item_to_save['types'] = {key: str(type(val)) for key, val in item_to_save.items()}
                            item_to_save['types']['types'] = str(type(item_to_save['types']))
                        except Exception as type_e:
                            print(f"\nWarning: Error generating 'types' dict for {item_to_save.get('id', 'unknown')}: {type_e}. Omitting.")
                            if 'types' in item_to_save: del item_to_save['types']

                        try:
                            serialized_data = pickle.dumps(item_to_save, protocol=pickle.HIGHEST_PROTOCOL)
                            buf = io.BytesIO()
                            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                                f.write(serialized_data)
                            compressed_value = buf.getvalue()
                            key = str(i).encode('utf-8')
                            put_success = txn.put(key, compressed_value, overwrite=False)
                            if not put_success:
                                raise RuntimeError(f'LMDB entry {i} in {out_path} already exists')
                            id_to_idx[item_to_save['id']] = i
                            lmdb_count += 1
                        except (pickle.PicklingError, TypeError) as pickle_e:
                            print(f"\nError pickling item {item_to_save.get('id', 'unknown')} (index {i}): {pickle_e}")
                            continue
                        except Exception as write_e:
                            print(f"\nError compressing/writing item {item_to_save.get('id', 'unknown')} (index {i}): {write_e}")
                            continue
                finally:
                    print(f"\nWriting LMDB metadata for {lmdb_count} items...")
                    txn.put(b'num_examples', str(lmdb_count).encode())
                    txn.put(b'serialization_format', serialization_format.encode())
                    try:
                        id_to_idx_serialized = pickle.dumps(id_to_idx, protocol=pickle.HIGHEST_PROTOCOL)
                        txn.put(b'id_to_idx', id_to_idx_serialized)
                    except Exception as meta_e:
                        print(f"Error serializing/writing LMDB metadata (id_to_idx): {meta_e}")
            env.close()
            print(f"LMDB writing complete. Wrote {lmdb_count} items to {out_path}")
        except lmdb.Error as lmdb_e:
            print(f"LMDB Error: {lmdb_e}")
            if isinstance(lmdb_e, lmdb.MapFullError): print("LMDB MapFullError: Increase map_size.")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"Error opening or writing LMDB: {e}")
            import traceback; traceback.print_exc()

    # --- Clean up DDP --- ## REMOVED ##
    # cleanup_ddp()


# Keep the if __name__ == '__main__': block
if __name__ == '__main__':
    # Multiprocessing start method - Keep as is
    try:
        current_context = torch.multiprocessing.get_start_method(allow_none=True)
        if current_context is None:
            required_torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            start_method = 'spawn' if required_torch_version >= (1, 7) else None
            if start_method:
                torch.multiprocessing.set_start_method(start_method, force=True)
    except Exception as e:
        print(f"Info: Could not set multiprocessing start method ('{e}'). Using default: '{torch.multiprocessing.get_start_method()}'.")
    main()