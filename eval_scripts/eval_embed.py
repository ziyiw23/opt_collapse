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
