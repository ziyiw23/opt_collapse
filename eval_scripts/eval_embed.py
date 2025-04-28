import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import lmdb
import pickle
import gzip
from tqdm import tqdm
import argparse
import os
import io
import random

# srun --partition=rbaltman \
#      --nodes=1 \
#      --ntasks=1 \
#      --cpus-per-task=4 \
#      --mem=64G \
#      --gres=gpu:1 \
#      --time=2:00:00 \
#      --output=dim_reduc.out \
#      --error=dim_reduc.err \
#     python ./eval_scripts/eval_embed.py --original_dir /scratch/groups/rbaltman/ziyiw23/ori_res/ --optimized_dir /scratch/groups/rbaltman/ziyiw23/opt_res/

plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.titlesize': 14
})

def load_lmdb_embeddings(lmdb_path):
    """Load and aggregate COLLAPSE embeddings while handling gzip compression and avoiding memory issues."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    embeddings = []
    metadata_list = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, desc=f"Loading LMDB entries from {lmdb_path}"):
            key_str = key.decode('utf-8', errors='replace')

            # Skip metadata entries
            if key_str in ['id_to_idx', 'num_examples', 'serialization_format']:
                continue

            try:
                # Decompress and deserialize
                decompressed = gzip.decompress(value)
                entry = pickle.loads(decompressed)

                if 'embeddings' in entry:
                    embeddings.append(entry['embeddings'].mean(axis=0))
                    metadata_list.append({'confidence': np.mean(entry.get('confidence', 0))})
            except Exception as e:
                print(f"Error processing {key_str}: {e}")

    embeddings = np.array(embeddings)
    metadata = pd.DataFrame(metadata_list)

    return embeddings, metadata


def plot_tsne(original_embeddings, original_metadata, optimized_embeddings, optimized_metadata, perplexity=30, n_samples=5000, output_path="/home/users/ziyiw23/COLLAPSE/embed_eval_output/t-sne_plot.png"):
    """t-SNE visualization comparing original and optimized embeddings"""
    
    # Sample embeddings if needed
    original_sample_size = min(n_samples, len(original_embeddings))
    optimized_sample_size = min(n_samples, len(optimized_embeddings))

    original_idx = np.random.choice(len(original_embeddings), original_sample_size, replace=False)
    optimized_idx = np.random.choice(len(optimized_embeddings), optimized_sample_size, replace=False)

    original_embeddings = original_embeddings[original_idx]
    optimized_embeddings = optimized_embeddings[optimized_idx]
    original_metadata = original_metadata.iloc[original_idx]
    optimized_metadata = optimized_metadata.iloc[optimized_idx]

    # Combine embeddings for joint dimensionality reduction
    all_embeddings = np.vstack([original_embeddings, optimized_embeddings])

    # Reduce dimensionality with PCA before t-SNE
    pca = PCA(n_components=min(50, all_embeddings.shape[1]))
    emb_pca = pca.fit_transform(all_embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, method='barnes_hut', random_state=42)
    emb_2d = tsne.fit_transform(emb_pca)
    
    # Split transformed embeddings back into original and optimized
    original_2d = emb_2d[:original_sample_size]
    optimized_2d = emb_2d[original_sample_size:]

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    
    plt.scatter(original_2d[:, 0], original_2d[:, 1], c='gold', label="Original", alpha=0.7, s=25, edgecolor='w', linewidth=0.5)
    
    plt.scatter(optimized_2d[:, 0], optimized_2d[:, 1], c='royalblue', label="Optimized", alpha=0.7, s=25, edgecolor='w', linewidth=0.5)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Projection of Protein Embeddings\nOriginal vs. Optimized")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved t-SNE plot to {output_path}")


def inspect_lmdb(lmdb_path, num_examples_to_show=3, show_array_shapes=True):
    """
    Inspects an LMDB dataset created by gen_embed.py or atom3d.

    Args:
        lmdb_path (str): Path to the LMDB directory (containing data.mdb).
        num_examples_to_show (int): How many specific examples to read and display.
        show_array_shapes (bool): Whether to print the shapes of numpy arrays found.
    """
    print(f"--- Inspecting LMDB: {lmdb_path} ---")

    if not os.path.isdir(lmdb_path):
        print(f"Error: LMDB directory not found at {lmdb_path}")
        return

    try:
        # Use subdir=True as atom3d datasets are directories
        # readonly=True is safer for inspection
        # lock=False might be necessary if the lock file is stale
        env = lmdb.open(lmdb_path, readonly=True, subdir=True, lock=False)
    except lmdb.Error as e:
        print(f"Error opening LMDB environment: {e}")
        return

    num_examples = 0
    id_to_idx_type = "Not Found"
    id_to_idx_sample = {}

    try:
        with env.begin() as txn:
            # Read metadata
            num_examples_bytes = txn.get(b'num_examples')
            if num_examples_bytes:
                num_examples = int(num_examples_bytes.decode())
                print(f"Metadata - num_examples: {num_examples}")
            else:
                print("Metadata - num_examples: Not found (may need to infer from cursor)")
                # Alternative way to estimate count if metadata key is missing
                num_examples = int(txn.stat()['entries']) - 3 # Subtract metadata keys usually
                print(f"Inferred count from entries: ~{num_examples}")


            serialization_format_bytes = txn.get(b'serialization_format')
            if serialization_format_bytes:
                serialization_format = serialization_format_bytes.decode()
                print(f"Metadata - serialization_format: {serialization_format}")
                if serialization_format != 'pkl':
                     print(f"Warning: Expected serialization format 'pkl', found '{serialization_format}'. Deserialization might fail.") # Fixed f-string
            else:
                print("Metadata - serialization_format: Not found (assuming 'pkl')")
                serialization_format = 'pkl' # Default assumption

            id_to_idx_bytes = txn.get(b'id_to_idx')
            if id_to_idx_bytes:
                try:
                    id_to_idx = pickle.loads(id_to_idx_bytes)
                    id_to_idx_type = type(id_to_idx)
                    if isinstance(id_to_idx, dict):
                         id_to_idx_sample = dict(list(id_to_idx.items())[:min(3, len(id_to_idx))])
                    print(f"Metadata - id_to_idx: Found (type: {id_to_idx_type}), Sample: {id_to_idx_sample}")
                except Exception as e:
                    print(f"Metadata - id_to_idx: Found but failed to deserialize: {e}")
            else:
                 print(f"Metadata - id_to_idx: Not Found")


            if num_examples == 0:
                print("No examples found based on metadata or inference.")
                return

            # Inspect a few examples
            indices_to_inspect = list(range(min(num_examples_to_show, num_examples)))
            if num_examples > num_examples_to_show:
                 # Add a random index if possible
                 try:
                      random_idx = random.randint(num_examples_to_show, num_examples - 1)
                      if random_idx not in indices_to_inspect:
                           indices_to_inspect.append(random_idx)
                 except ValueError: # Handle case where num_examples <= num_examples_to_show
                      pass

            print(f"\nInspecting entries at indices: {indices_to_inspect}...")

            for i in indices_to_inspect:
                print(f"\n--- Entry Index: {i} ---")
                key = str(i).encode('utf-8')
                compressed_value = txn.get(key)

                if compressed_value is None:
                    print("  Error: Entry not found.")
                    continue

                try:
                    # Decompress and deserialize
                    buf = io.BytesIO(compressed_value)
                    with gzip.GzipFile(fileobj=buf, mode='rb') as f:
                        data = pickle.load(f)

                    print(f"  Keys: {list(data.keys())}")

                    # Print details of some keys, especially array shapes
                    if 'id' in data: print(f"  id: {data['id']}")
                    if 'pooling_type' in data: print(f"  pooling_type: {data['pooling_type']}")

                    if show_array_shapes:
                        for k, v in data.items():
                            if isinstance(v, np.ndarray):
                                print(f"  {k}: numpy array, shape={v.shape}, dtype={v.dtype}")
                            elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                                # Handle lists of arrays if they weren't stacked (shouldn't happen with current gen_embed)
                                try:
                                     stacked = np.stack(v)
                                     print(f"  {k}: list of numpy arrays, stacked shape={stacked.shape}, first dtype={v[0].dtype}")
                                except ValueError:
                                     print(f"  {k}: list of numpy arrays (cannot stack), count={len(v)}, first shape={v[0].shape}, first dtype={v[0].dtype}")
                            elif isinstance(v, (list, dict)) and k != 'types':
                                 print(f"  {k}: {type(v).__name__}, length/size={len(v)}")


                except pickle.UnpicklingError as e:
                    print(f"  Error deserializing entry {i}: {e}")
                except gzip.BadGzipFile as e:
                    print(f"  Error decompressing entry {i}: {e}")
                except Exception as e:
                    print(f"  Unexpected error processing entry {i}: {e}")

    except lmdb.Error as e:
        print(f"Error during LMDB transaction: {e}")
    finally:
        if 'env' in locals() and env:
            env.close()
            print("\n--- LMDB Inspection Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', type=str, required=True, help="Path to the original LMDB directory")
    parser.add_argument('--optimized_dir', type=str, required=True, help="Path to the optimized LMDB directory")
    args = parser.parse_args()

    if not os.path.exists(args.original_dir):
        raise FileNotFoundError(f"Original LMDB file not found: {args.original_dir}")

    if not os.path.exists(args.optimized_dir):
        raise FileNotFoundError(f"Optimized LMDB file not found: {args.optimized_dir}")

    print("Loading original embeddings...")
    original_embeddings, original_metadata = load_lmdb_embeddings(args.original_dir)
    
    print("Loading optimized embeddings...")
    optimized_embeddings, optimized_metadata = load_lmdb_embeddings(args.optimized_dir)
    
    print("Plotting t-SNE comparison...")
    plot_tsne(original_embeddings, original_metadata, optimized_embeddings, optimized_metadata)
