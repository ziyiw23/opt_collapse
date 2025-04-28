# V2.5 - Refactored for Pickling

import numpy as np
import os
import argparse
import torch
import pandas as pd
from atom3d.datasets import load_dataset # Keep this
import atom3d.util.file as fi
from collapse import initialize_model, atom_info # Assuming these are correct
# from atom3d.filters.filters import first_model_filter # Moved to utils
import collections as col
import random
# import torch_cluster # Only needed if BaseTransformCPU uses it
from torch_geometric.data import Batch, Data # Batch needed for collate
from torch.utils.data import Dataset, DataLoader
import lmdb # For manual LMDB writing
import pickle # For LMDB serialization
from tqdm import tqdm # Progress bar
import gzip # For compression
import io
# from scipy.spatial import KDTree # Moved to utils

import time
import sys

# --- Import from utils --- ## ADDED ##
from embedding_utils import (
    BaseTransform, GraphPreparationTransformCPU, TransformedDatasetWrapper,
    graph_collate_fn # Assuming collate fn is also moved or adaptable
)

# --- Seeding and Constants ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Deterministic takes priority

# ELEMENT_MAPPING, DEFAULT_ELEMENT Moved to embedding_utils.py
# --- Helper Functions (_normalize, _rbf, _edge_features) Moved to embedding_utils.py ---
# --- BaseTransform (CPU Version for Workers) Moved to embedding_utils.py ---
# --- sample_functional_center Moved to embedding_utils.py ---
# --- extract_env_for_residue (CPU version using SciPy KDTree) Moved to embedding_utils.py ---
# --- prepare_graphs_for_protein (Worker Task Helper - CPU) Moved to embedding_utils.py ---
# --- Graph Preparation Transform (CPU Version for Workers) Moved to embedding_utils.py ---
# --- Custom Collate Function (Robust Version) ---
# Moved to embedding_utils.py or keep here if needed
# def graph_collate_fn(batch): ...

# --- Dataset Wrapper (Applies CPU Transform in Worker) Moved to embedding_utils.py ---


# --- is_valid_pdb ---
# (Keep as before)
def is_valid_pdb(filepath):
    try: return os.path.getsize(filepath) > 0
    except OSError: return False

# --- main function (V2.5 - Using utils) --- ## MODIFIED ##
def main():
    parser = argparse.ArgumentParser(description="V2.5 Embedding generation with Utils Refactor") # Updated desc
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
    parser.add_argument('--num_workers', type=int, default=9, help="Number of DataLoader workers")
    parser.add_argument('--batch_size', type=int, default=1, help="Number of *proteins* per GPU batch")
    parser.add_argument('--prefetch_factor', type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using num_workers: {args.num_workers}, batch_size: {args.batch_size}")

    # --- Load Model (Main Process - GPU) ---
    print("Loading model...")
    model = initialize_model(args.checkpoint, device=device)
    model.eval()
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        try:
            model = torch.compile(model, mode="default")
            print("Model compiled successfully.")
        except Exception as e: print(f"Warning: Model compilation failed: {e}")

    # --- Load RAW Dataset ---
    print(f"Loading RAW dataset structure from: {args.data_dir} with filetype: {args.filetype}")
    try:
        raw_dataset_base = load_dataset(args.data_dir, args.filetype, transform=None)
        dataset_len = len(raw_dataset_base)
        print(f"Initial raw dataset size: {dataset_len}")
        if dataset_len == 0: sys.exit("Error: Loaded raw dataset is empty.")
    except Exception as e: sys.exit(f"Error loading raw dataset: {e}")

    # --- Instantiate the CPU Transform (defined in utils) ---
    graph_transform_cpu = GraphPreparationTransformCPU(
         include_hets=args.include_hets,
         env_radius=args.env_radius,
         max_neighbors=args.max_neighbors
    )

    # --- Wrap Dataset with Transform (defined in utils) ---
    dataset_for_loader = TransformedDatasetWrapper(raw_dataset_base, graph_transform_cpu)

    # --- Splitting Logic (Applied to the final dataset object) ---
    indices = np.arange(len(dataset_for_loader))
    if args.num_splits > 1:
        if args.split_id < 1 or args.split_id > args.num_splits:
             sys.exit(f"Error: split_id ({args.split_id}) must be between 1 and {args.num_splits}")
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

    # --- Setup DataLoader ---
    dataloader = DataLoader(
        final_dataset_to_load,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_collate_fn, # Use collate_fn from utils
        pin_memory=True if str(device) != 'cpu' else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,
        # persistent_workers=True if args.num_workers > 0 else False
    )

    # --- Main Processing & Inference Loop with OOM Fallback ---
    print("Starting parallel processing and LMDB writing...")
    start_loop_time = time.time()
    results_to_save = [] # Collect final dictionaries {id: ..., embeddings: ..., etc.}
    processed_protein_count = 0
    failed_protein_count = 0 # Proteins completely failed even after fallback
    processed_residue_count = 0

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing Batches")):

        if batch_data is None: continue # Skip if collation failed

        graph_batch_cpu, metadata_batch = batch_data # Keep graphs on CPU initially for fallback

        if graph_batch_cpu is None or metadata_batch is None: continue

        # Track unique protein IDs in this specific batch
        proteins_in_batch_ids = list(set(m['protein_id'] for m in metadata_batch))
        graphs_on_cpu_list = graph_batch_cpu.to_data_list() # Keep CPU list for potential fallback

        # --- Attempt Normal Batch Inference ---
        final_embs_np = None
        batch_failed_oom = False
        try:
            # Move graph batch to GPU for main attempt
            graph_batch_gpu = graph_batch_cpu.to(device)

            with torch.no_grad():
                with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                    embs, _ = model.online_encoder(graph_batch_gpu, return_projection=False)
                    final_embs_np = embs.float().cpu().numpy()
                processed_residue_count += final_embs_np.shape[0]

            # Clear GPU memory quickly after successful inference
            del graph_batch_gpu
            del embs
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                batch_failed_oom = True # Set flag to trigger fallback
                print(f"\nOOM on batch {batch_idx}. Batch size: {args.batch_size}, Num graphs: {len(graphs_on_cpu_list)}. Triggering fallback...")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                # Ensure potential partial results are cleared
                final_embs_np = None
                # We will handle processing below in the fallback section
            else:
                # Other runtime errors - fail the whole batch
                print(f"\nRuntime error during inference on batch {batch_idx}: {e}. Failing proteins: {proteins_in_batch_ids}")
                failed_protein_count += len(proteins_in_batch_ids)
                continue # Skip to next batch
        except Exception as e:
            # Other non-runtime errors during inference
            print(f"\nNon-runtime error during inference on batch {batch_idx}: {e}. Failing proteins: {proteins_in_batch_ids}")
            failed_protein_count += len(proteins_in_batch_ids)
            continue # Skip batch

        # --- Process Results (Normal or Fallback) ---
        results_this_batch = col.defaultdict(lambda: col.defaultdict(list))

        if not batch_failed_oom:
            # --- Normal Processing: Distribute embeddings ---
            if final_embs_np is None: # Should only happen on non-OOM error above
                print(f"\nWarning: Embeddings None after inference (Batch {batch_idx}). Skipping.")
                failed_protein_count += len(proteins_in_batch_ids)
                continue

            if len(metadata_batch) != final_embs_np.shape[0]:
                print(f"\nCRITICAL WARNING: Mismatch! Metadata len ({len(metadata_batch)}) != Embeddings len ({final_embs_np.shape[0]}) for batch {batch_idx}. Skipping batch.")
                failed_protein_count += len(proteins_in_batch_ids)
                continue

            # Group results by protein ID
            for i, meta in enumerate(metadata_batch):
                protein_id = meta['protein_id']
                results_this_batch[protein_id]['resids'].append(meta['resid'])
                results_this_batch[protein_id]['chains'].append(meta['chain'])
                results_this_batch[protein_id]['confidence'].append(meta['confidence'])
                results_this_batch[protein_id]['embeddings'].append(final_embs_np[i])

        else:
            # --- OOM Fallback Processing: Chunked Inference + Mean Pooling ---
            print(f"--- Starting OOM Fallback for Batch {batch_idx} ---")
            # Group graphs and metadata by protein ID first
            graphs_by_protein = col.defaultdict(list)
            metadata_by_protein = col.defaultdict(list)
            for graph, meta in zip(graphs_on_cpu_list, metadata_batch):
                graphs_by_protein[meta['protein_id']].append(graph)
                metadata_by_protein[meta['protein_id']].append(meta)

            for protein_id in proteins_in_batch_ids: # Iterate through proteins that were in the failed batch
                protein_graphs = graphs_by_protein[protein_id]
                protein_metadata = metadata_by_protein[protein_id]
                num_residues = len(protein_graphs)
                chunk_size = 250
                protein_embeddings_list = []
                protein_failed_chunking = False
                # print(f"  Processing {protein_id} ({num_residues} residues) in chunks...")

                for i in range(0, num_residues, chunk_size):
                    chunk_graphs = protein_graphs[i : i + chunk_size]
                    if not chunk_graphs: continue

                    try:
                        chunk_batch = Batch.from_data_list(chunk_graphs).to(device)
                        with torch.no_grad():
                            with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                                chunk_embs, _ = model.online_encoder(chunk_batch, return_projection=False)
                                protein_embeddings_list.append(chunk_embs.float().cpu().numpy())
                        del chunk_batch
                        del chunk_embs
                        # Minimal clearing within chunk loop if memory is extremely tight
                        # if torch.cuda.is_available(): torch.cuda.empty_cache()

                    except RuntimeError as chunk_e:
                        if "CUDA out of memory" in str(chunk_e):
                            print(f"  ❌ OOM even during chunking fallback for {protein_id} (chunk starting at {i}). Protein cannot be processed.")
                        else:
                            print(f"  ❌ Runtime error during fallback chunk for {protein_id}: {chunk_e}")
                        protein_failed_chunking = True
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        break # Stop processing chunks for this protein
                    except Exception as chunk_e:
                        print(f"  ❌ Non-runtime error during fallback chunk for {protein_id}: {chunk_e}")
                        protein_failed_chunking = True
                        break

                # After processing all chunks for a protein...
                if not protein_failed_chunking and protein_embeddings_list:
                    try:
                        all_protein_embs = np.concatenate(protein_embeddings_list, axis=0)
                        # Check if number of embeddings matches number of graphs processed
                        if all_protein_embs.shape[0] == num_residues:
                            # --- Store Per-Residue Embeddings (like normal case) --- ## MODIFIED ##
                            # We now store the full per-residue embeddings even in fallback
                            results_this_batch[protein_id]['embeddings'] = [all_protein_embs] # Store 2D array in list
                            results_this_batch[protein_id]['pooling_type'] = ['chunked_oom_fallback'] # Flag how it was generated
                            # Keep metadata from all residues
                            results_this_batch[protein_id]['resids'] = [m['resid'] for m in protein_metadata]
                            results_this_batch[protein_id]['chains'] = [m['chain'] for m in protein_metadata]
                            results_this_batch[protein_id]['confidence'] = [m['confidence'] for m in protein_metadata]
                            print(f"  ✓ Fallback successful for {protein_id} (Stored {all_protein_embs.shape[0]} residue embeddings).") # Updated message
                        else:
                            print(f"  ❌ Mismatch after fallback concatenation for {protein_id}: {all_protein_embs.shape[0]} vs {num_residues}. Skipping.")
                            protein_failed_chunking = True # Mark as failed
                    except Exception as store_e: # Changed exception name
                        print(f"  ❌ Error during storing fallback result for {protein_id}: {store_e}") # Updated message
                        protein_failed_chunking = True

                if protein_failed_chunking:
                    failed_protein_count += 1 # Increment total fail count

            print(f"--- Finished OOM Fallback for Batch {batch_idx} ---")
            # Clear cache once after all fallbacks for the batch
            if torch.cuda.is_available(): torch.cuda.empty_cache()


        # --- Finalize and Add Results from Batch ---
        # This loop now handles results from both normal processing and successful fallbacks
        for protein_id, data in results_this_batch.items():
            if data.get('embeddings'): # Check if embeddings key exists and is not empty
                embedding_list = data['embeddings']
                embedding_array = None

                if isinstance(embedding_list, list) and embedding_list:
                    if len(embedding_list) > 1: # Normal case: List of per-residue embeddings
                        try:
                            # Check if all elements are numpy arrays before stacking
                            if all(isinstance(emb, np.ndarray) for emb in embedding_list):
                                embedding_array = np.stack(embedding_list, axis=0)
                            else:
                                print(f"\nWarning: Non-numpy array found in embedding list for {protein_id}. Skipping.")
                                failed_protein_count += 1
                                continue
                        except Exception as stack_e:
                            print(f"\nError stacking embeddings for {protein_id} (normal path): {stack_e}")
                            failed_protein_count += 1
                            continue # Skip this protein

                    elif len(embedding_list) == 1: # OOM Fallback case: List containing one 2D array
                         # Check if the single element is a numpy array
                         if isinstance(embedding_list[0], np.ndarray):
                              embedding_array = embedding_list[0]
                         else:
                              print(f"\nWarning: Single element in embedding list for {protein_id} is not a numpy array. Type: {type(embedding_list[0])}. Skipping.")
                              failed_protein_count += 1
                              continue
                # else: Empty list or not a list
                #     pass # Let the next check handle None

                # --- Proceed if we have a valid 2D embedding_array --- 
                if embedding_array is not None:
                     data['embeddings'] = embedding_array # Store the final 2D array

                     # Check if the number of metadata items matches the first dimension of the embedding array
                     num_meta_items = len(data.get('resids', []))
                     if num_meta_items == embedding_array.shape[0]:
                          results_to_save.append({'id': protein_id, **data})
                          processed_protein_count += 1
                          # Accumulate residue count based on successful processing
                          processed_residue_count += embedding_array.shape[0] # This count was wrong before
                     else:
                          print(f"\nFinal internal mismatch for {protein_id} before saving. Metadata count ({num_meta_items}) != Embeddings dim 0 ({embedding_array.shape[0]}). Skipping.")
                          # Avoid double counting if already marked failed during fallback
                          if not data.get('pooling_type', None) or 'chunked_oom_fallback' not in data['pooling_type']:
                               failed_protein_count +=1 # Only count failure here if not already counted in fallback
                else:
                    # This handles cases where the list was empty, not a list, or stacking failed
                    print(f"\nWarning: Could not obtain valid embedding array for {protein_id}. Skipping.")
                    # Avoid double counting if already marked failed
                    if protein_id not in (p_id for p_id, d in results_this_batch.items() if d.get('pooling_type')): # Check if not already failed fallback
                        failed_protein_count += 1
            # else: # Handles cases where embeddings key might be missing or empty initially
                # Failure potentially already counted during fallback attempt or initial inference error
                pass

# --- End Main Loop ---

    # --- End Main Loop ---
    loop_end_time = time.time()
    print("-" * 30)
    print(f"Embedding generation loop finished in {loop_end_time - start_loop_time:.2f} seconds.")
    print(f"Successfully processed and collected results for: {processed_protein_count} proteins.")
    print(f"Total residues processed (embeddings generated): {processed_residue_count}")
    if failed_protein_count > 0:
        print(f"Failed/Skipped proteins (OOM or other error): {failed_protein_count}")
    print("-" * 30)

    # --- Write results to LMDB ---
    print(f"Writing {len(results_to_save)} results to LMDB directory: {out_path}")
    if not results_to_save:
        print("No results to save.")
        return

    # atom3d uses a large default map size and subdir=True
    map_size = int(1e13) # 10 TB (atom3d default) - adjust if needed, but large is safer
    serialization_format = 'pkl' # Explicitly using pickle

    try:
        # Use subdir=True to create a directory like atom3d does
        env = lmdb.open(out_path, map_size=map_size, subdir=True) # Changed subdir to True

        id_to_idx = {}
        final_count = 0
        with env.begin(write=True) as txn:
            for i, result_dict in enumerate(tqdm(results_to_save, desc="Writing LMDB")):
                # Add the 'types' dictionary like atom3d
                result_dict['types'] = {key: str(type(val)) for key, val in result_dict.items()}
                result_dict['types']['types'] = str(type(result_dict['types']))

                try:
                    # Use pickle for serialization_format='pkl'
                    # No need to force protocol usually
                    serialized_data = pickle.dumps(result_dict)

                    # Compress using gzip
                    buf = io.BytesIO()
                    # Use compresslevel=6 like atom3d
                    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                       f.write(serialized_data)
                    compressed_value = buf.getvalue()

                    # Use sequential integer index 'i' as the key (encoded)
                    key = str(i).encode('utf-8')
                    # Write to LMDB
                    put_success = txn.put(key, compressed_value, overwrite=False)
                    if not put_success:
                        # This should not happen if we iterate sequentially with 'i'
                        print(f"Error: LMDB key {i} already exists in {out_path}. This indicates an issue.")
                        # Decide how to handle: overwrite or raise error? Atom3d raises.
                        raise RuntimeError(f'LMDB entry {i} in {out_path} already exists')

                    # Store mapping from original ID to index key 'i'
                    id_to_idx[result_dict['id']] = i
                    final_count += 1 # Increment count only on successful write

                except Exception as write_e:
                    print(f"Error pickling/compressing/writing key {result_dict['id']} (index {i}) to LMDB: {write_e}")
                    # Decide if you want to skip this item or stop entirely
                    continue # Skip faulty item

            # --- Write Metadata (within the same transaction) ---
            print(f"Writing LMDB metadata for {final_count} items...")
            txn.put(b'num_examples', str(final_count).encode())
            txn.put(b'serialization_format', serialization_format.encode())
            # Serialize the id_to_idx mapping itself using pickle
            id_to_idx_serialized = pickle.dumps(id_to_idx)
            txn.put(b'id_to_idx', id_to_idx_serialized)

        env.close()
        print(f"LMDB writing complete. Wrote {final_count} items.")

    except Exception as e:
        print(f"Error opening or writing main transaction to LMDB: {e}")
        import traceback; traceback.print_exc()
        # Clean up potentially partially created directory if needed
        # import shutil
        # if os.path.exists(out_path):
        #     print(f"Cleaning up potentially corrupted LMDB directory: {out_path}")
        #     shutil.rmtree(out_path)


if __name__ == '__main__':
     required_torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
     start_method = 'spawn' if required_torch_version >= (1, 7) else None
     try:
         current_context = torch.multiprocessing.get_start_method(allow_none=True)
         if current_context is None:
              if start_method:
                  torch.multiprocessing.set_start_method(start_method, force=True)
                  print(f"Set multiprocessing start method to '{start_method}'.")
         elif current_context != start_method and start_method is not None:
              print(f"Warning: Multiprocessing context already set to '{current_context}', attempting to force '{start_method}'.")
              torch.multiprocessing.set_start_method(start_method, force=True)
     except RuntimeError as e:
          print(f"Info: Could not set multiprocessing start method ('{e}'). Using default: '{torch.multiprocessing.get_start_method()}'.")
          pass
     except ValueError as e:
          print(f"Info: Default multiprocessing start method likely sufficient ('{e}').")
          pass

     main()