# V2.5 - Refactored for Pickling and Modularity

import numpy as np
import os
import argparse
import torch
import pandas as pd
from atom3d.datasets import load_dataset 
import atom3d.util.file as fi
from collapse import initialize_model, atom_info # Keep initialize_model here for now
import collections as col
import random
from torch_geometric.data import Batch, Data 
from torch.utils.data import Dataset, DataLoader
import lmdb 
import pickle 
from tqdm import tqdm 
import gzip
import io
import time
import sys

# Import necessary components from the utils module
from embedding_utils import (
    GraphPreparationTransformCPU, TransformedDatasetWrapper,
    graph_collate_fn
)

# --- Seeding --- 
# (Keep seeding near the top or move to a setup function if preferred)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Helper Functions --- 

def _parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="V2.5 Embedding generation with Utils Refactor & Modularity")
    parser.add_argument('data_dir', type=str, help="Directory containing raw data (e.g., PDBs).")
    parser.add_argument('out_dir', type=str, help="Directory to save the output LMDB database.")
    parser.add_argument('--split_id', type=int, default=0, help="Split ID (1 to num_splits) to process. 0 means process all.")
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt', help="Path to model checkpoint.")
    parser.add_argument('--filetype', type=str, default='pdb', help="Input file type (e.g., pdb, cif).")
    parser.add_argument('--num_splits', type=int, default=1, help="Number of splits to divide the dataset into.")
    parser.add_argument('--env_radius', type=float, default=10.0, help="Environment radius for graph construction.")
    parser.add_argument('--include_hets', action='store_true', default=False, help="Include heteroatoms in processing.")
    # parser.add_argument('--max_neighbors', type=int, default=32, help="Max neighbors (Not currently used by radius_graph setup)") # Argument removed as it's unused
    parser.add_argument('--compile_model', action='store_true', help="Enable torch.compile (PyTorch 2.0+).")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of DataLoader workers.") # Default adjusted slightly
    parser.add_argument('--batch_size', type=int, default=1, help="Number of *proteins* per GPU batch.")
    parser.add_argument('--prefetch_factor', type=int, default=2, help="DataLoader prefetch factor.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (potentially more verbose output).")
    return parser.parse_args()

def _setup_device():
    """Sets up and returns the device (CUDA or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def _load_and_prepare_model(args, device):
    """Loads, prepares (eval mode, optional compilation), and returns the model."""
    print("Loading model...")
    model = initialize_model(args.checkpoint, device=device) # initialize_model is from collapse
    model.eval()
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        try:
            model = torch.compile(model, mode="default")
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")
    return model

def _prepare_dataloader(args):
    """Loads raw data, sets up transforms/wrapping, handles splits, returns DataLoader."""
    print(f"Loading RAW dataset structure from: {args.data_dir} with filetype: {args.filetype}")
    try:
        # Load the base dataset without transforms initially
        raw_dataset_base = load_dataset(args.data_dir, args.filetype, transform=None)
        dataset_len = len(raw_dataset_base)
        print(f"Initial raw dataset size: {dataset_len}")
        if dataset_len == 0:
            sys.exit("Error: Loaded raw dataset is empty.")
    except Exception as e:
        sys.exit(f"Error loading raw dataset: {e}")

    # Instantiate the CPU Transform (defined in embedding_utils)
    # Note: max_neighbors is removed as BaseTransform doesn't use it directly now
    graph_transform_cpu = GraphPreparationTransformCPU(
         include_hets=args.include_hets,
         env_radius=args.env_radius
         # num_rbf can be added here if needed, defaults to 16
    )

    # Wrap the raw dataset with the transform (defined in embedding_utils)
    dataset_for_loader = TransformedDatasetWrapper(raw_dataset_base, graph_transform_cpu)

    # Apply splitting logic if requested
    indices = np.arange(len(dataset_for_loader))
    if args.num_splits > 1:
        if args.split_id < 1 or args.split_id > args.num_splits:
             sys.exit(f"Error: split_id ({args.split_id}) must be between 1 and {args.num_splits}")
        split_indices = np.array_split(indices, args.num_splits)[args.split_id - 1]
        if len(split_indices) == 0:
            print(f"Warning: Split {args.split_id} has 0 examples.")
        print(f'Processing split {args.split_id}/{args.num_splits} with {len(split_indices)} examples...')
        final_dataset_to_load = torch.utils.data.Subset(dataset_for_loader, split_indices)
    else:
        print(f'Processing full dataset with {len(dataset_for_loader)} examples...')
        final_dataset_to_load = dataset_for_loader

    if len(final_dataset_to_load) == 0:
        print("Warning: Final dataset to process is empty after splitting. Exiting.")
        sys.exit(0) # Graceful exit if nothing to process

    # Setup DataLoader
    dataloader = DataLoader(
        final_dataset_to_load,
        batch_size=args.batch_size,
        shuffle=False, # Typically False for inference/embedding generation
        num_workers=args.num_workers,
        collate_fn=graph_collate_fn, # Use collate_fn from embedding_utils
        pin_memory=torch.cuda.is_available(), # Pin memory if using GPU
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,
        persistent_workers=True if args.num_workers > 0 else False # Use persistent workers if possible
    )
    print(f"DataLoader configured with: num_workers={args.num_workers}, batch_size={args.batch_size}")
    return dataloader

def _run_inference_and_collect(model, dataloader, device, batch_size):
    """Runs inference loop, handles OOM fallback, collects results."""
    print("Starting parallel processing and inference loop...")
    start_loop_time = time.time()
    results_to_save = [] # Collect final dictionaries {id: ..., embeddings: ..., etc.}
    processed_protein_count = 0
    failed_protein_count = 0 # Proteins completely failed even after fallback
    processed_residue_count = 0

    oom_fallback_active = False # Flag to track if fallback was used

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing Batches")):

        if batch_data is None:
            # print(f"Skipping batch {batch_idx} due to collation failure.") # Optional debug
            continue # Skip if collation failed

        graph_batch_cpu, metadata_batch = batch_data # Keep graphs on CPU initially

        if graph_batch_cpu is None or metadata_batch is None:
            # print(f"Skipping batch {batch_idx} due to empty batch data after collation.") # Optional debug
            continue

        # Track unique protein IDs in this specific batch
        proteins_in_batch_ids = list(set(m['protein_id'] for m in metadata_batch))
        graphs_on_cpu_list = graph_batch_cpu.to_data_list() # Keep CPU list for potential fallback

        # --- Attempt Normal Batch Inference ---
        final_embs_np = None
        batch_failed_oom = False
        try:
            graph_batch_gpu = graph_batch_cpu.to(device) # Move graph batch to target device

            with torch.no_grad():
                # Use autocast for potential speedup on CUDA with float16 support
                with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                    embs, _ = model.online_encoder(graph_batch_gpu, return_projection=False)
                    final_embs_np = embs.float().cpu().numpy() # Move back to CPU, ensure float32
                    # processed_residue_count += final_embs_np.shape[0] # Count moved to after validation

            # Clear GPU memory quickly after successful inference
            del graph_batch_gpu, embs
            if str(device) != 'cpu': torch.cuda.empty_cache()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                batch_failed_oom = True
                oom_fallback_active = True # Mark that fallback was needed at least once
                print(f"\nOOM on batch {batch_idx}. Batch size: {batch_size}, Num graphs: {len(graphs_on_cpu_list)}. Triggering fallback...")
                if str(device) != 'cpu': torch.cuda.empty_cache()
                final_embs_np = None # Ensure embeddings are cleared
                # Fallback processing happens below
            else:
                print(f"\nRuntime error during inference on batch {batch_idx}: {e}. Failing proteins: {proteins_in_batch_ids}")
                failed_protein_count += len(proteins_in_batch_ids)
                continue # Skip to next batch
        except Exception as e:
            print(f"\nNon-runtime error during inference on batch {batch_idx}: {e}. Failing proteins: {proteins_in_batch_ids}")
            failed_protein_count += len(proteins_in_batch_ids)
            continue # Skip batch

        # --- Process Results (Normal or Fallback) ---
        results_this_batch = col.defaultdict(lambda: col.defaultdict(list))

        if not batch_failed_oom:
            # --- Normal Processing: Distribute embeddings ---
            if final_embs_np is None:
                print(f"\nWarning: Embeddings None after successful inference (Batch {batch_idx}). Skipping batch.")
                failed_protein_count += len(proteins_in_batch_ids)
                continue

            if len(metadata_batch) != final_embs_np.shape[0]:
                print(f"\nCRITICAL WARNING: Mismatch! Metadata len ({len(metadata_batch)}) != Embeddings len ({final_embs_np.shape[0]}) for batch {batch_idx}. Skipping batch.")
                failed_protein_count += len(proteins_in_batch_ids)
                continue

            # Group results by protein ID
            current_emb_idx = 0
            for meta in metadata_batch:
                protein_id = meta['protein_id']
                results_this_batch[protein_id]['resids'].append(meta['resid'])
                results_this_batch[protein_id]['chains'].append(meta['chain'])
                results_this_batch[protein_id]['confidence'].append(meta['confidence'])
                results_this_batch[protein_id]['embeddings'].append(final_embs_np[current_emb_idx])
                current_emb_idx += 1

        else: # batch_failed_oom is True
            # --- OOM Fallback Processing: Chunked Inference --- 
            print(f"--- Starting OOM Fallback Processing for Batch {batch_idx} ---")
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
                chunk_size = 250 # Process residues in smaller chunks
                protein_embeddings_list = []
                protein_failed_chunking = False
                # print(f"  Fallback: Processing {protein_id} ({num_residues} residues) in chunks...")

                for i in range(0, num_residues, chunk_size):
                    chunk_graphs = protein_graphs[i : i + chunk_size]
                    if not chunk_graphs: continue

                    try:
                        chunk_batch = Batch.from_data_list(chunk_graphs).to(device)
                        with torch.no_grad():
                            with torch.autocast(device_type=str(device.type), dtype=torch.float16, enabled=(str(device.type) == 'cuda')):
                                chunk_embs, _ = model.online_encoder(chunk_batch, return_projection=False)
                                protein_embeddings_list.append(chunk_embs.float().cpu().numpy())
                        del chunk_batch, chunk_embs
                        # Minimal clearing inside chunk loop if memory is extremely tight
                        # if str(device) != 'cpu': torch.cuda.empty_cache()

                    except RuntimeError as chunk_e:
                        if "CUDA out of memory" in str(chunk_e):
                            print(f"  ❌ OOM even during chunking fallback for {protein_id} (chunk starting at {i}). Protein cannot be processed.")
                        else:
                            print(f"  ❌ Runtime error during fallback chunk for {protein_id}: {chunk_e}")
                        protein_failed_chunking = True
                        if str(device) != 'cpu': torch.cuda.empty_cache()
                        break # Stop processing chunks for this protein
                    except Exception as chunk_e:
                        print(f"  ❌ Non-runtime error during fallback chunk for {protein_id}: {chunk_e}")
                        protein_failed_chunking = True
                        break

                # After processing all chunks for a protein...
                if not protein_failed_chunking and protein_embeddings_list:
                    try:
                        # Concatenate chunk results into a single 2D array for the protein
                        all_protein_embs = np.concatenate(protein_embeddings_list, axis=0)
                        if all_protein_embs.shape[0] == num_residues:
                            # Store the per-residue embeddings like the normal case
                            # Put the single 2D array into the list for consistent structure
                            results_this_batch[protein_id]['embeddings'] = [all_protein_embs] 
                            results_this_batch[protein_id]['pooling_type'] = ['chunked_oom_fallback'] # Flag how generated
                            # Store all corresponding metadata
                            results_this_batch[protein_id]['resids'] = [m['resid'] for m in protein_metadata]
                            results_this_batch[protein_id]['chains'] = [m['chain'] for m in protein_metadata]
                            results_this_batch[protein_id]['confidence'] = [m['confidence'] for m in protein_metadata]
                            # print(f"  ✓ Fallback successful for {protein_id}. Stored {all_protein_embs.shape[0]} residue embeddings.")
                        else:
                            print(f"  ❌ Mismatch after fallback concatenation for {protein_id}: {all_protein_embs.shape[0]} vs {num_residues}. Skipping protein.")
                            protein_failed_chunking = True
                    except Exception as store_e:
                        print(f"  ❌ Error storing/concatenating fallback result for {protein_id}: {store_e}")
                        protein_failed_chunking = True

                if protein_failed_chunking:
                    failed_protein_count += 1 # Increment total fail count

            # print(f"--- Finished OOM Fallback Processing for Batch {batch_idx} ---")
            if str(device) != 'cpu': torch.cuda.empty_cache() # Clear cache after all fallbacks for batch

        # --- Finalize and Aggregate Results from Batch --- 
        # Consolidate results (handles both normal and successful fallback cases)
        for protein_id, data in results_this_batch.items():
            if 'embeddings' not in data or not data['embeddings']:
                # This protein likely failed during chunking, already counted
                # print(f"Debug: Skipping {protein_id}, no valid embeddings found in results_this_batch.")
                 continue

            embedding_list = data['embeddings']
            embedding_array = None

            # Try to form the final 2D numpy array
            if isinstance(embedding_list, list) and embedding_list:
                if len(embedding_list) > 1: # Normal case: List of per-residue embeddings
                    if all(isinstance(emb, np.ndarray) for emb in embedding_list):
                        try: embedding_array = np.stack(embedding_list, axis=0) 
                        except ValueError as stack_e: 
                             print(f"\nError stacking normal embeddings for {protein_id}: {stack_e}. Sizes: {[e.shape for e in embedding_list]}")
                             # Failure count handled below if embedding_array is None
                    # else: print(f"Warning: Non-numpy array in normal embedding list for {protein_id}")
                elif len(embedding_list) == 1: # OOM Fallback case: List containing one 2D array
                     if isinstance(embedding_list[0], np.ndarray):
                          embedding_array = embedding_list[0]
                     # else: print(f"Warning: Fallback embedding element not np array for {protein_id}")

            # Check if we have a valid 2D array and if dimensions match metadata
            if embedding_array is not None:
                num_meta_items = len(data.get('resids', []))
                if num_meta_items == embedding_array.shape[0]:
                      data['embeddings'] = embedding_array # Replace list with final array
                      results_to_save.append({'id': protein_id, **data})
                      processed_protein_count += 1
                      processed_residue_count += embedding_array.shape[0] # Count successful residues
                else:
                    print(f"\nFinal validation mismatch for {protein_id}. Meta count ({num_meta_items}) != Embeddings dim 0 ({embedding_array.shape[0]}). Skipping protein.")
                    # Avoid double counting failures from OOM fallback
                    if not data.get('pooling_type', []) or 'chunked_oom_fallback' not in data.get('pooling_type', []):
                           failed_protein_count += 1
                    else:
                        print(f"\nFinal validation mismatch for {protein_id}. Meta count ({num_meta_items}) != Embeddings dim 0 ({embedding_array.shape[0]}). Skipping protein.")
                        # Avoid double counting failures from OOM fallback
                        if not data.get('pooling_type', []) or 'chunked_oom_fallback' not in data.get('pooling_type', []):
                            failed_protein_count += 1 # Count failure only if not already counted during OOM fail

# --- End Main Loop ---
    loop_end_time = time.time()
    print("-" * 30)
    print(f"Embedding generation loop finished in {loop_end_time - start_loop_time:.2f} seconds.")
    print(f"Successfully processed and collected results for: {processed_protein_count} proteins.")
    print(f"Total residues processed (embeddings generated): {processed_residue_count}")
    if failed_protein_count > 0:
        print(f"Failed/Skipped proteins (OOM or other error): {failed_protein_count}")
    if oom_fallback_active:
         print("Note: OOM fallback mechanism was activated during processing.")
    print("-" * 30)
    return results_to_save

def _write_results_to_lmdb(results_to_save, out_path):
    """Writes the collected results to an LMDB database."""
    print(f"Writing {len(results_to_save)} results to LMDB directory: {out_path}")
    if not results_to_save:
        print("No results to save.")
        return

    os.makedirs(out_path, exist_ok=True) # Ensure base directory exists

    # Use large map size for LMDB; atom3d uses subdir=True which creates a dir
    map_size = int(1e13) # 10 TB
    serialization_format = 'pkl'
    lmdb_target_path = os.path.join(out_path) # Path for lmdb.open

    try:
        # Use subdir=True to create a directory structure like atom3d
        # If the directory already exists from a previous run, lmdb.open might fail
        # Consider removing the target dir first if overwrite is intended
        env = lmdb.open(lmdb_target_path, map_size=map_size, subdir=True, create=True) # Added create=True

        id_to_idx = {}
        final_count = 0
        with env.begin(write=True) as txn:
            for i, result_dict in enumerate(tqdm(results_to_save, desc="Writing LMDB")):
                protein_id = result_dict.get('id', f'unknown_{i}')
                try:
                    # Add the 'types' dictionary like atom3d
                    # Ensure 'embeddings' exists before creating types dict
                    if 'embeddings' in result_dict:
                        result_dict['types'] = {key: str(type(val)) for key, val in result_dict.items()}
                        result_dict['types']['types'] = str(type(result_dict['types'])) # Add type of types dict itself
                    else: # Handle case where embeddings might somehow be missing
                        print(f"Warning: Missing 'embeddings' key for {protein_id} before writing. Skipping item.")
                        continue
                        
                    serialized_data = pickle.dumps(result_dict)
                    # Compress using gzip
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                       f.write(serialized_data)
                    compressed_value = buf.getvalue()

                    key = str(i).encode('utf-8')
                    put_success = txn.put(key, compressed_value, overwrite=False)
                    if not put_success:
                        raise RuntimeError(f'LMDB entry {i} in {out_path} already exists')

                    id_to_idx[protein_id] = i
                    final_count += 1

                except Exception as write_e:
                    print(f"Error pickling/compressing/writing ID {protein_id} (index {i}): {write_e}")
                    continue # Skip faulty item

            # Write Metadata
            print(f"Writing LMDB metadata for {final_count} items...")
            txn.put(b'num_examples', str(final_count).encode())
            txn.put(b'serialization_format', serialization_format.encode())
            id_to_idx_serialized = pickle.dumps(id_to_idx)
            txn.put(b'id_to_idx', id_to_idx_serialized)

        env.close()
        print(f"LMDB writing complete. Wrote {final_count} items to {lmdb_target_path}.")

    except Exception as e:
        print(f"Error opening or writing main transaction to LMDB at {lmdb_target_path}: {e}")
        import traceback; traceback.print_exc()
        # Consider cleanup logic here if needed

def set_multiprocessing_start_method():
    """Sets the multiprocessing start method, preferably to 'spawn'."""
    # required_torch_version = tuple(map(int, torch.__version__.split('.')[:2])) # Commented out as unused
    start_method = 'spawn' # Generally safer
    try:
        current_context = torch.multiprocessing.get_start_method(allow_none=True)
        if current_context is None:
            if start_method: # Check if start_method is actually set (it is here)
                torch.multiprocessing.set_start_method(start_method, force=True)
                print(f"Set multiprocessing start method to '{start_method}'.")
        elif current_context != start_method: # Changed elif to else, corrected indentation
            print(f"Warning: Multiprocessing context already '{current_context}'. Forcing '{start_method}'.")
            torch.multiprocessing.set_start_method(start_method, force=True)
    except Exception as e: # Correctly placed except block
        print(f"Info: Could not set multiprocessing start method ('{e}'). Using default: '{torch.multiprocessing.get_start_method()}'.")

def main():
    """Main execution function."""
    args = _parse_arguments()
    device = _setup_device()
    set_multiprocessing_start_method()
    
    model = _load_and_prepare_model(args, device)
    dataloader = _prepare_dataloader(args)
    
    results = _run_inference_and_collect(model, dataloader, device, args.batch_size)
    
    # Determine final output path based on splitting
    out_path = args.out_dir
    if args.num_splits > 1:
         # Ensure the split subdirectory exists within the main out_dir
         out_path = os.path.join(args.out_dir, f'embeddings_split_{args.split_id}')
         # LMDB writing function will create the final dir (like 'embeddings_split_1/data.mdb')
    else:
        # LMDB writing function will create the 'data.mdb' etc inside args.out_dir
        pass 
        
    _write_results_to_lmdb(results, out_path)

    print("Embedding generation script finished.")


if __name__ == '__main__':
     main()