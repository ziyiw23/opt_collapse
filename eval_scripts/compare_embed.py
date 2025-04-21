import lmdb
import pickle
import gzip
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
from scipy.spatial import procrustes

# List of evaluation metrics (Procrustes will now be computed globally)
eval_metrics = ['cosine', 'flipped_cosine', 'pcc', 'l2_diff', 'mse', 'spearman']

def inspect_lmdb(lmdb_dir, output_file="inspect_output.txt", limit=10):
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with open(output_file, "w", encoding="utf-8") as f:
        with env.begin() as txn:
            cursor = txn.cursor()
            for i, (key, value) in enumerate(cursor):
                key_str = key.decode('utf-8', errors='replace')
                if key_str in ['id_to_idx', 'num_examples', 'serialization_format']:
                    f.write(f"Skipping metadata key: {key_str}\n")
                    continue
                f.write(f"Key: {key_str}\n")
                f.write(f"Type of value: {type(value)}\n")
                try:
                    decompressed = gzip.decompress(value)
                    data = pickle.loads(decompressed)
                    f.write(f"Value: {data}\n")
                    if isinstance(data, pd.DataFrame):
                        csv_filename = f"{key_str}_dataframe.csv"
                        data.to_csv(csv_filename, index=False)
                        f.write(f"DataFrame saved to {csv_filename}\n")
                except Exception as e:
                    f.write(f"Error processing value: {e}\n")
                f.write("-" * 40 + "\n")
                if i >= limit:
                    break

def get_common_diff_keys(original_dir, optimized_dir, out_dir):
    """Get intersection of numeric keys from both databases"""
    def numeric_keys(env):
        with env.begin() as txn:
            return {k.decode() for k in txn.cursor().iternext(values=False) if k.decode().isdigit()}
    
    orig_env = lmdb.open(original_dir, readonly=True, lock=False)
    opt_env = lmdb.open(optimized_dir, readonly=True, lock=False)
    
    orig_keys = numeric_keys(orig_env)
    opt_keys = numeric_keys(opt_env)
    
    orig_env.close()
    opt_env.close()
    
    print(f"Original DB has {len(orig_keys)} numeric entries")
    print(f"Optimized DB has {len(opt_keys)} numeric entries")
    
    common = sorted(orig_keys & opt_keys)
    print(f"Found {len(common)} common numeric entries")

    # Different keys
    only_in_orig = sorted(orig_keys - opt_keys)
    only_in_opt = sorted(opt_keys - orig_keys)

    print(f"Found {len(only_in_orig)} keys only in the original DB")
    print(f"Found {len(only_in_opt)} keys only in the optimized DB")

    if len(only_in_orig) > 0:
        with open(os.path.join(out_dir, "only_in_orig.txt"), "w") as f:
            for key in only_in_orig:
                f.write(f"{key}\n")
    if len(only_in_opt) > 0:
        with open(os.path.join(out_dir, "only_in_opt.txt"), "w") as f:
            for key in only_in_opt:
                f.write(f"{key}\n")

    return common

def get_embedding(txn, key):
    """Extract and average 2D embeddings to 1D"""
    try:
        value = txn.get(key.encode())
        if not value:
            return None
        data = pickle.loads(gzip.decompress(value))
        emb_2d = data['embeddings'].astype(np.float32)
        return emb_2d.mean(axis=0)  # Shape becomes (512,)
    except Exception as e:
        print(f"Error processing {key}: {e}")
        return None

def compute_metrics(orig_emb, opt_emb, centroid):
    """
    For two 1D embeddings and a centroid (computed over centered embeddings),
    compute the following metrics:
      - cosine similarity between centered embeddings
      - cosine similarity after flipping the optimized embedding
      - Pearson correlation coefficient (pcc) between raw embeddings
      - L2 norm difference between raw embeddings
      - Mean Squared Error (MSE) between raw embeddings
      - Spearman's rank correlation coefficient between raw embeddings
    """
    # Center both embeddings
    orig_centered = orig_emb - centroid
    opt_centered = opt_emb - centroid

    norm_orig = orig_centered / (np.linalg.norm(orig_centered) + 1e-8)
    norm_opt = opt_centered / (np.linalg.norm(opt_centered) + 1e-8)
    
    cosine = np.dot(norm_orig, norm_opt)
    flipped_cosine = np.dot(norm_orig, -norm_opt)
    
    # Pearson correlation on raw embeddings
    pcc, _ = scipy.stats.pearsonr(orig_emb.flatten(), opt_emb.flatten())
    
    # L2 norm difference between raw embeddings
    l2_diff = np.linalg.norm(orig_emb - opt_emb)
    
    # Mean Squared Error (MSE) between raw embeddings
    mse = np.mean((orig_emb - opt_emb) ** 2)
    
    # Spearman's rank correlation coefficient on raw embeddings
    spearman_corr, _ = scipy.stats.spearmanr(orig_emb.flatten(), opt_emb.flatten())
    
    return {
        'cosine': cosine,
        'flipped_cosine': flipped_cosine,
        'pcc': pcc,
        'l2_diff': l2_diff,
        'mse': mse,
        'spearman': spearman_corr
    }

def compute_centroid(orig_env, opt_env, common_keys):
    """Compute centroid from 1D embeddings across all common keys"""
    total_sum = None
    count = 0
    
    with orig_env.begin() as orig_txn, opt_env.begin() as opt_txn:
        for key in tqdm(common_keys, desc="Computing centroid"):
            orig_emb = get_embedding(orig_txn, key)
            opt_emb = get_embedding(opt_txn, key)
            if orig_emb is None or opt_emb is None:
                continue
            if orig_emb.shape != opt_emb.shape:
                print(f"Shape mismatch in {key}: {orig_emb.shape} vs {opt_emb.shape}")
                continue
            vec_sum = orig_emb + opt_emb
            if total_sum is None:
                total_sum = np.zeros_like(vec_sum)
            total_sum += vec_sum
            count += 2
    
    return total_sum / count if count else None

def compute_all_metrics(orig_env, opt_env, common_keys, centroid):
    """Compute metrics for each key and return a dictionary and the embedding matrices."""
    metrics = {}
    all_orig = []
    all_opt = []

    with orig_env.begin() as orig_txn, opt_env.begin() as opt_txn:
        for key in tqdm(common_keys, desc="Computing all metrics"):
            orig_emb = get_embedding(orig_txn, key)
            opt_emb = get_embedding(opt_txn, key)
            if orig_emb is None or opt_emb is None:
                continue
            if orig_emb.shape != opt_emb.shape:
                continue
            met = compute_metrics(orig_emb, opt_emb, centroid)
            metrics[key] = met
            all_orig.append(orig_emb)
            all_opt.append(opt_emb)

    return metrics, np.stack(all_orig), np.stack(all_opt)

def print_summary_statistics(metrics, metric_name):
    """Print summary statistics for a given metric from the metrics dict."""
    values = np.array([v[metric_name] for v in metrics.values()])
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    print(f"Summary Statistics for {metric_name}:")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Standard Deviation: {std_val:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")

def plot_similarity_distribution(opt_metrics, metric_name, output_dir, two_run_metrics=None):
    """Generate an overlaid histogram for a given metric."""
    opt_values = [v[metric_name] for v in opt_metrics.values()]
    plt.figure(figsize=(10, 6))
    
    if two_run_metrics is not None:
        two_run_values = [v[metric_name] for v in two_run_metrics.values()]
        plt.hist(opt_values, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Optimized vs Original')
        plt.hist(two_run_values, bins=50, alpha=0.3, color='red', edgecolor='black', label='Two Originals')
    else:
        plt.hist(opt_values, bins=50, alpha=0.75, color='steelblue', edgecolor='black', label='Optimized')

    plt.title(f"{metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    
    hist_path = os.path.join(output_dir, f"{metric_name}_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Histogram saved to {hist_path}")

# Sanity check functions remain unchanged
def validate_self_similarity(env, keys):
    with env.begin() as txn:
        for key in keys[:5]:
            emb = get_embedding(txn, key)
            if emb is None:
                continue
            sim = np.dot(emb / np.linalg.norm(emb), emb / np.linalg.norm(emb))
            if not np.isclose(sim, 1.0, atol=1e-6):
                print(f"⚠️ Self-similarity failed for {key}: {sim:.4f}")
            else:
                print(f"✅ Self-similarity valid for {key}: {sim:.4f}")

def validate_centroid(env, keys, centroid):
    with env.begin() as txn:
        emb_sum = np.zeros_like(centroid)
        count = 0
        for key in keys[:100]:
            emb = get_embedding(txn, key)
            if emb is not None:
                emb_sum += emb
                count += 1
        calculated_centroid = emb_sum / count
        diff = np.abs(centroid - calculated_centroid).mean()
        print(f"Centroid validation - Mean difference: {diff:.2e}")

def check_embedding_consistency(orig_env, opt_env, keys):
    with orig_env.begin() as orig_txn, opt_env.begin() as opt_txn:
        for key in keys[:5]:
            orig = get_embedding(orig_txn, key)
            opt = get_embedding(opt_txn, key)
            if orig is None or opt is None:
                continue
            if not np.allclose(orig, opt, atol=1e-6):
                print(f"⚠️ Embedding mismatch for {key}")
            else:
                print(f"✅ Embeddings identical for {key}")

def delete_files_in_directory(directory_path):
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
        print("Old output dir files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient embedding comparison with additional metrics")
    parser.add_argument('--original', required=True, help="Original LMDB directory")
    parser.add_argument('--optimized', required=True, help="Optimized LMDB directory")
    parser.add_argument('--out_dir', default="/home/users/ziyiw23/COLLAPSE/embed_eval_output", help="Output directory")
    parser.add_argument('--second_ori', default=None, help="Second original LMDB directory for comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    delete_files_in_directory(args.out_dir)
    
    common_keys = get_common_diff_keys(args.original, args.optimized, args.out_dir)
    if not common_keys:
        print("No common keys found!")
        return
    
    orig_env = lmdb.open(args.original, readonly=True, lock=False)
    opt_env = lmdb.open(args.optimized, readonly=True, lock=False)
    
    centroid = compute_centroid(orig_env, opt_env, common_keys)
    if centroid is None:
        print("Failed to compute centroid")
        return

    print("\nComputing All Metrics...")
    metrics, orig_mat, opt_mat = compute_all_metrics(orig_env, opt_env, common_keys, centroid)
    opt_env.close()

    # Compute global Procrustes once over all embeddings
    mtx1, mtx2, global_disparity = procrustes(orig_mat, opt_mat)

    print(""" 
  ____  __ __  __  __  __  __   ____  _____ __  __   
 (_ (_`|  |  ||  \/  ||  \/  | / () \ | () )\ \/ /   
.__)__) \___/ |_|\/|_||_|\/|_|/__/\__\|_|\_\ |__|    
    """)

    # Print summary statistics for each metric
    for metric in eval_metrics:
        print(f"\n=== Summary for {metric} ===")
        print_summary_statistics(metrics, metric)
    print(f"✅ Global Procrustes disparity: {global_disparity:.6f}")

    # Plot similarity distributions
    second_metrics = None
    if args.second_ori is not None:
        second_orig_env = lmdb.open(args.second_ori, readonly=True, lock=False)
        second_common_keys = get_common_diff_keys(args.original, args.second_ori, args.out_dir)
        centroid_sec = compute_centroid(orig_env, second_orig_env, second_common_keys)
        second_metrics, _, _ = compute_all_metrics(orig_env, second_orig_env, second_common_keys, centroid_sec)
        second_orig_env.close()
        print(""" 
  ____  __ __  __  __  __  __   ____  _____ __  __   
 (_ (_`|  |  ||  \/  ||  \/  | / () \ | () )\ \/ /   
.__)__) \___/ |_|\/|_||_|\/|_|/__/\__\|_|\_\ |__|    
        """)
        for metric in eval_metrics:
            print(f"\n=== Summary for {metric} (second original) ===")
            print_summary_statistics(second_metrics, metric)
    orig_env.close()
    
    for metric in eval_metrics:
        print(f"\n⏳ Plotting {metric} Distribution")
        plot_similarity_distribution(metrics, metric, args.out_dir, two_run_metrics=second_metrics)
    
    # Visualize global alignment using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj1 = pca.fit_transform(mtx1)
    proj2 = pca.transform(mtx2)
    plt.figure(figsize=(10, 6))
    plt.scatter(proj1[:, 0], proj1[:, 1], label="Original", alpha=0.5)
    plt.scatter(proj2[:, 0], proj2[:, 1], label="Optimized (aligned)", alpha=0.5)
    plt.legend()
    plt.title("Global Embedding Alignment after Procrustes")
    pca_path = os.path.join(args.out_dir, "Procrustes_alignment.png")
    plt.savefig(pca_path)
    plt.close()
    print(f"\n✅ PCA alignment visualization saved to {pca_path}")
    
    output_path = os.path.join(args.out_dir, "all_metrics.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✅ All metrics saved to {output_path}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"✅ Evaluation completed in {end_time - start_time:.2f} seconds.")
