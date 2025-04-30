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
import pprint
import io
import random
from typing import Tuple, Dict, List, Any, Optional
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# python eval_scripts/compare_embed.py --original /scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res_1000 --optimized /scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res_1000

# List of evaluation metrics for residue-level comparison
eval_metrics = ['cosine', 'l2_diff', 'mse'] # Removed pcc, spearman as they are less meaningful for single vectors

# --- Constants and Style Configuration ---
DEFAULT_N_SAMPLES: int = 5000
DEFAULT_PCA_COMPONENTS: int = 50
DEFAULT_TSNE_PERPLEXITY: int = 30
RANDOM_STATE: int = 42
DEFAULT_OUTPUT_FILENAME: str = "t-sne_comparison_plot.png"
OUTPUT_DIR: str = "/home/users/ziyiw23/COLLAPSE/embed_eval_output/" # Define output dir

# Matplotlib/Seaborn styling
# plt.style.use('seaborn-v0_8-darkgrid') # Example of a modern style
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    """Get intersection of numeric keys (representing proteins/entries) from both databases"""
    def numeric_keys(env):
        with env.begin() as txn:
            # Check for id_to_idx first, which maps protein IDs to numeric keys
            id_map_bytes = txn.get(b'id_to_idx')
            if id_map_bytes:
                id_map = pickle.loads(id_map_bytes)
                # Return the protein IDs which are the logical keys
                return set(id_map.keys())
            else:
                # Fallback: assume numeric keys directly represent proteins if id_to_idx is missing
                print("Warning: id_to_idx not found in LMDB. Falling back to numeric keys.")
                return {k.decode() for k in txn.cursor().iternext(values=False) if k.decode().isdigit()}

    orig_env = lmdb.open(original_dir, readonly=True, lock=False)
    opt_env = lmdb.open(optimized_dir, readonly=True, lock=False)

    orig_keys = numeric_keys(orig_env)
    opt_keys = numeric_keys(opt_env)

    orig_env.close()
    opt_env.close()

    print(f"Original DB has {len(orig_keys)} protein entries (from id_to_idx or numeric keys)")
    print(f"Optimized DB has {len(opt_keys)} protein entries (from id_to_idx or numeric keys)")

    common = sorted(list(orig_keys & opt_keys))
    print(f"Found {len(common)} common protein entries")

    # Different keys
    only_in_orig = sorted(list(orig_keys - opt_keys))
    only_in_opt = sorted(list(opt_keys - orig_keys))

    print(f"Found {len(only_in_orig)} keys only in the original DB")
    print(f"Found {len(only_in_opt)} keys only in the optimized DB")

    if len(only_in_orig) > 0:
        with open(os.path.join(out_dir, "only_in_orig_proteins.txt"), "w") as f:
            for key in only_in_orig:
                f.write(f"{key}\n")
    if len(only_in_opt) > 0:
        with open(os.path.join(out_dir, "only_in_opt_proteins.txt"), "w") as f:
            for key in only_in_opt:
                f.write(f"{key}\n")

    return common # Returns list of common protein IDs

def get_protein_data(txn, protein_id_or_numeric_key, id_map=None):
    """Extracts the full data dictionary for a protein entry."""
    try:
        # Determine the actual LMDB key (numeric index)
        if id_map:
            if protein_id_or_numeric_key not in id_map:
                print(f"Error: Protein ID {protein_id_or_numeric_key} not found in id_map.")
                return None
            numeric_key = str(id_map[protein_id_or_numeric_key])
        else:
            # Assume the key provided is already the numeric key
            numeric_key = str(protein_id_or_numeric_key)

        value = txn.get(numeric_key.encode())
        if not value:
            # print(f"Warning: No value found for key {numeric_key} (Protein: {protein_id_or_numeric_key}).")
            return None
        # Decompress and unpickle
        decompressed_value = gzip.decompress(value)
        data = pickle.loads(decompressed_value)

        # Validate expected keys
        if not isinstance(data, dict) or 'embeddings' not in data or 'chains' not in data or 'resids' not in data:
             print(f"Warning: Invalid data format for key {numeric_key}. Missing required keys.")
             # print(f"Data: {data}") # Optional: Print problematic data
             return None

        # Ensure embeddings are numpy array
        if not isinstance(data['embeddings'], np.ndarray):
             print(f"Warning: Embeddings are not a numpy array for key {numeric_key}.")
             return None

        # Ensure metadata lists have same length as embeddings first dimension
        num_res = data['embeddings'].shape[0]
        if not (len(data['chains']) == num_res and len(data['resids']) == num_res):
            print(f"Warning: Metadata length mismatch for key {numeric_key}. Embeddings: {num_res}, Chains: {len(data['chains'])}, Resids: {len(data['resids'])}.")
            return None # Skip inconsistent entries

        # Cast embeddings to float32 for consistency
        data['embeddings'] = data['embeddings'].astype(np.float32)
        return data # Return the full dictionary

    except lmdb.Error as e:
        print(f"LMDB Error processing key {protein_id_or_numeric_key}: {e}")
        return None
    except gzip.BadGzipFile:
        print(f"Error: Bad Gzip data for key {protein_id_or_numeric_key}. Not gzipped?")
        return None
    except pickle.UnpicklingError as e:
        print(f"Error unpickling data for key {protein_id_or_numeric_key}: {e}")
        return None
    except Exception as e:
        print(f"General Error processing key {protein_id_or_numeric_key}: {type(e).__name__}: {e}")
        import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return None

def compute_residue_metrics(orig_emb_1d, opt_emb_1d):
    """
    Computes metrics between two individual 1D residue embedding vectors.
    Args:
        orig_emb_1d (np.ndarray): 1D numpy array for original embedding.
        opt_emb_1d (np.ndarray): 1D numpy array for optimized embedding.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    if orig_emb_1d is None or opt_emb_1d is None or orig_emb_1d.shape != opt_emb_1d.shape or orig_emb_1d.ndim != 1:
        return {metric: np.nan for metric in eval_metrics} # Return NaNs if input invalid

    # Cosine Similarity
    norm_orig = np.linalg.norm(orig_emb_1d)
    norm_opt = np.linalg.norm(opt_emb_1d)
    if norm_orig < 1e-8 or norm_opt < 1e-8: # Avoid division by zero for zero vectors
        cosine = 1.0 if norm_orig < 1e-8 and norm_opt < 1e-8 else 0.0
    else:
        cosine = np.dot(orig_emb_1d, opt_emb_1d) / (norm_orig * norm_opt)

    # L2 Norm Difference
    l2_diff = np.linalg.norm(orig_emb_1d - opt_emb_1d)

    # Mean Squared Error (MSE)
    mse = np.mean((orig_emb_1d - opt_emb_1d) ** 2)

    return {
        'cosine': cosine,
        'l2_diff': l2_diff,
        'mse': mse
    }


def compute_all_residue_metrics(orig_env, opt_env, common_protein_keys):
    """Computes metrics for each corresponding residue across all common proteins."""
    all_residue_metrics = [] # List to store metric dicts for each residue pair
    all_orig_residue_embs = []
    all_opt_residue_embs = []
    processed_protein_count = 0
    processed_residue_count = 0
    skipped_proteins = 0
    skipped_residues = 0

    # Load id_to_idx maps once
    with orig_env.begin() as orig_txn, opt_env.begin() as opt_txn:
        orig_id_map_bytes = orig_txn.get(b'id_to_idx')
        opt_id_map_bytes = opt_txn.get(b'id_to_idx')
        orig_id_map = pickle.loads(orig_id_map_bytes) if orig_id_map_bytes else None
        opt_id_map = pickle.loads(opt_id_map_bytes) if opt_id_map_bytes else None
        if orig_id_map is None or opt_id_map is None:
             print("Warning: Could not load id_to_idx map from one or both DBs. Assuming common_keys are numeric LMDB keys.")

    with orig_env.begin() as orig_txn, opt_env.begin() as opt_txn:
        for protein_key in tqdm(common_protein_keys, desc="Computing residue metrics"):#
            orig_data = get_protein_data(orig_txn, protein_key, orig_id_map)
            opt_data = get_protein_data(opt_txn, protein_key, opt_id_map)

            if orig_data is None or opt_data is None:
                skipped_proteins += 1
                continue # Skip protein if data retrieval failed for either

            orig_embeddings = orig_data['embeddings'] # Should be (num_res, embed_dim)
            opt_embeddings = opt_data['embeddings']

            # Create residue identifier -> index map for both
            orig_res_map = {(c, r): i for i, (c, r) in enumerate(zip(orig_data['chains'], orig_data['resids']))}
            opt_res_map = {(c, r): i for i, (c, r) in enumerate(zip(opt_data['chains'], opt_data['resids']))}

            common_residues = sorted(list(orig_res_map.keys() & opt_res_map.keys()))

            if not common_residues:
                # print(f"Warning: No common residues found for protein {protein_key}.")
                skipped_proteins += 1 # Count as skipped if no residues overlap
                continue

            protein_processed_flag = False
            for chain_resid_tuple in common_residues:
                try:
                    orig_idx = orig_res_map[chain_resid_tuple]
                    opt_idx = opt_res_map[chain_resid_tuple]

                    orig_emb_1d = orig_embeddings[orig_idx]
                    opt_emb_1d = opt_embeddings[opt_idx]

                    # Compute metrics for this residue pair
                    residue_met = compute_residue_metrics(orig_emb_1d, opt_emb_1d)

                    # Add identifiers to the metric dict
                    residue_met['protein_id'] = protein_key
                    residue_met['chain'] = chain_resid_tuple[0]
                    residue_met['resid'] = chain_resid_tuple[1]

                    all_residue_metrics.append(residue_met)

                    # Collect aligned embeddings for global Procrustes/PCA
                    all_orig_residue_embs.append(orig_emb_1d)
                    all_opt_residue_embs.append(opt_emb_1d)
                    processed_residue_count += 1
                    protein_processed_flag = True

                except Exception as e:
                     print(f"Error processing residue {chain_resid_tuple} in protein {protein_key}: {e}")
                     skipped_residues += 1

            if protein_processed_flag:
                 processed_protein_count += 1
            else:
                 # If we iterated common_residues but failed for all of them
                 skipped_proteins += 1

    print(f"Residue Metric Calculation Summary:")
    print(f" Successfully processed proteins: {processed_protein_count}")
    print(f" Total processed residue pairs: {processed_residue_count}")
    print(f" Skipped proteins (data load fail or no common residues): {skipped_proteins}")
    print(f" Skipped individual residues (error during metric calc): {skipped_residues}")

    # Convert collected embeddings to large NumPy arrays
    orig_residue_matrix = np.stack(all_orig_residue_embs) if all_orig_residue_embs else np.array([])
    opt_residue_matrix = np.stack(all_opt_residue_embs) if all_opt_residue_embs else np.array([])

    return all_residue_metrics, orig_residue_matrix, opt_residue_matrix

def print_summary_statistics(residue_metrics_list, metric_name):
    """Print summary statistics for a given metric from the list of residue metrics."""
    # Filter out potential NaNs from failed metric calculations
    values = np.array([m[metric_name] for m in residue_metrics_list if not np.isnan(m[metric_name])])

    if values.size == 0:
        print(f"Summary Statistics for {metric_name}: No valid data.")
        return

    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    q25, q75 = np.percentile(values, [25, 75])

    print(f"Summary Statistics for {metric_name} (across {len(values)} residues):")
    print(f"  Mean:   {mean_val:.6f}")
    print(f"  Median: {median_val:.6f}")
    print(f"  Std Dev:{std_val:.6f}")
    print(f"  Min:    {min_val:.6f}")
    print(f"  Max:    {max_val:.6f}")
    print(f"  25%:    {q25:.6f}")
    print(f"  75%:    {q75:.6f}")

def plot_similarity_distribution(residue_metrics_list, metric_name, output_dir):
    """Generate a histogram for a given residue-level metric."""
    # Filter out potential NaNs
    values = [m[metric_name] for m in residue_metrics_list if not np.isnan(m[metric_name])]

    if not values:
        print(f"Cannot plot {metric_name}: No valid data.")
        return

    plt.figure(figsize=(10, 6))

    plt.hist(values, bins=50, alpha=0.75, color='steelblue', edgecolor='black')

    plt.title(f"Residue-Level {metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    # plt.legend() # No legend needed for single histogram

    hist_path = os.path.join(output_dir, f"residue_{metric_name}_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Residue histogram saved to {hist_path}")

# Sanity check functions can be removed or adapted if needed for residue-level
# e.g., check self-similarity for a few residues, but less critical now.

def delete_files_in_directory(directory_path):
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
        print("Old output dir files deleted successfully.")
    except OSError as e:
        print(f"Error occurred while deleting files: {e}")

# --- NEW INSPECTION FUNCTION --- (Keep as is, useful for debugging LMDB structure)
def inspect_first_embeddings(lmdb_dir, limit=5):
    """
    Inspects and prints the content of the first few data entries in an LMDB database.

    Args:
        lmdb_dir (str): Path to the LMDB database directory.
        limit (int): Number of entries to inspect.
    """
    print(f"\n--- Inspecting first {limit} data entries from {lmdb_dir} ---")
    inspected_count = 0

    try:
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    except lmdb.Error as e:
        print(f"Error opening LMDB {lmdb_dir}: {e}")
        return

    # Get id_map if it exists
    id_map = None
    with env.begin() as txn:
        id_map_bytes = txn.get(b'id_to_idx')
        if id_map_bytes:
            try:
                id_map = pickle.loads(id_map_bytes)
                print("  (Using id_to_idx map for inspection)")
            except Exception as e:
                print(f"  Warning: Failed to load id_to_idx map: {e}")

    with env.begin() as txn:
        cursor = txn.cursor()
        for key_bytes, value in cursor:
            if inspected_count >= limit:
                break

            key = key_bytes.decode('utf-8')

            # Use the key directly if it's numeric OR if id_map is missing
            if not key.isdigit() and id_map is not None:
                # Skip non-numeric keys if we have an id_map (like num_examples)
                # print(f"  Skipping non-data key: {key}")
                continue

            protein_id_label = key # Default label
            if id_map:
                # Find protein ID corresponding to this numeric key
                found_id = None
                for p_id, num_key_idx in id_map.items():
                    if str(num_key_idx) == key:
                        found_id = p_id
                        break
                if found_id:
                    protein_id_label = f"{found_id} (LMDB key: {key})"
                else:
                    protein_id_label = f"Unknown Protein (LMDB key: {key})"


            print(f"\n--- Entry: {protein_id_label} ---")
            try:
                # Decompress and deserialize
                data = pickle.loads(gzip.decompress(value))

                print(f"  Data Type: {type(data)}")
                if isinstance(data, dict):
                    print("  Content:")
                    # Use pprint for better readability
                    data_to_print = {}
                    for k, v in data.items():
                        if k == 'embeddings' and isinstance(v, np.ndarray):
                            data_to_print[k] = f"Numpy Array (shape: {v.shape}, dtype: {v.dtype}) - First 5: {v.flatten()[:5]}..."
                        elif isinstance(v, list) and len(v) > 5:
                             data_to_print[k] = f"List (len: {len(v)}) - First 5: {v[:5]}..."
                        else:
                             data_to_print[k] = v
                    pprint.pprint(data_to_print, indent=4)
                else:
                    print(f"  Content: {data}")

                inspected_count += 1

            except gzip.BadGzipFile:
                print(f"  Error decompressing value for key {key}. Is it gzipped?")
            except pickle.UnpicklingError:
                 print(f"  Error unpickling value for key {key}.")
            except Exception as e:
                print(f"  Unexpected error inspecting key {key}: {e}")

    env.close()
    if inspected_count == 0:
        print("  No valid data entries found to inspect within the key range checked.")
    print(f"\n--- Finished inspecting {inspected_count} entries ---")

def main():
    parser = argparse.ArgumentParser(description="Residue-level embedding comparison")
    parser.add_argument('--original', required=True, help="Original LMDB directory")
    parser.add_argument('--optimized', required=True, help="Optimized LMDB directory")
    parser.add_argument('--out_dir', default=OUTPUT_DIR, help="Output directory")
    # Remove --second_ori argument as comparing two originals at residue level is less common
    # parser.add_argument('--second_ori', default=None, help="Second original LMDB directory for comparison")
    parser.add_argument('--inspect', action='store_true', help="Inspect first few embeddings data structure from both DBs.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    delete_files_in_directory(args.out_dir)

    if args.inspect:
        inspect_first_embeddings(args.original, limit=3)
        inspect_first_embeddings(args.optimized, limit=3)
        print("\nInspection complete. Continuing with comparison...")

    common_protein_keys = get_common_diff_keys(args.original, args.optimized, args.out_dir)
    if not common_protein_keys:
        print("No common protein keys found! Exiting.")
        return

    orig_env = lmdb.open(args.original, readonly=True, lock=False, readahead=False) # Disable readahead for potentially better random access
    opt_env = lmdb.open(args.optimized, readonly=True, lock=False, readahead=False)

    print("\nComputing All Residue Metrics...")
    # This function now returns residue metrics list, and the aligned residue matrices
    residue_metrics, orig_residue_mat, opt_residue_mat = compute_all_residue_metrics(orig_env, opt_env, common_protein_keys)

    orig_env.close() # Close environments after computation
    opt_env.close()

    if not residue_metrics or orig_residue_mat.size == 0:
         print("No valid residue pairs found for comparison. Exiting.")
         return

    # Compute global Procrustes on the residue matrices
    print("\nPerforming Global Procrustes Analysis on Residue Embeddings...")
    mtx1, mtx2, global_disparity = procrustes(orig_residue_mat, opt_residue_mat)

    print("""
______ _____ _____ _   _ _    _____ 
| ___ \  ___/  ___| | | | |  |_   _|
| |_/ / |__ \ `--.| | | | |    | |  
|    /|  __| `--. \ | | | |    | |  
| |\ \| |___/\__/ / |_| | |____| |  
\_| \_\____/\____/ \___/\_____/\_/  
""")

    # Print summary statistics for each residue-level metric
    for metric in eval_metrics:
        print(f"\n=== Summary for Residue-Level {metric} ===")
        print_summary_statistics(residue_metrics, metric)
    print(f"\n✅ Global Residue Procrustes disparity: {global_disparity:.6f}")

    # Plot residue-level similarity distributions
    for metric in eval_metrics:
        print(f"\n⏳ Plotting Residue-Level {metric} Distribution")
        plot_similarity_distribution(residue_metrics, metric, args.out_dir)

    # Visualize global alignment using PCA on Procrustes-aligned residue embeddings
    print("\n⏳ Generating PCA plot of aligned residue embeddings...")
    try:
        n_components_pca = min(DEFAULT_PCA_COMPONENTS, mtx1.shape[0], mtx1.shape[1]) # Adjust components if fewer residues/dims
        if n_components_pca < 2:
            print(" Not enough data points or dimensions for 2D PCA plot.")
        else:
            pca = PCA(n_components=n_components_pca)
            proj1 = pca.fit_transform(mtx1) # Fit on original aligned
            proj2 = pca.transform(mtx2)     # Transform optimized aligned
            
            # Sample points for plotting if too many residues
            num_residues_to_plot = min(len(proj1), DEFAULT_N_SAMPLES)
            indices = np.random.choice(len(proj1), num_residues_to_plot, replace=False)
            
            plt.figure(figsize=(12, 8))
            plt.scatter(proj1[indices, 0], proj1[indices, 1], label="Original Residues (Aligned)", alpha=0.5, s=10)
            plt.scatter(proj2[indices, 0], proj2[indices, 1], label="Optimized Residues (Aligned)", alpha=0.5, s=10)
            plt.legend()
            plt.title(f"Residue Embedding Alignment after Procrustes (PCA, {num_residues_to_plot} samples)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)
            pca_path = os.path.join(args.out_dir, "Residue_Procrustes_PCA_alignment.png")
            plt.savefig(pca_path)
            plt.close()
            print(f"✅ PCA alignment visualization saved to {pca_path}")
    except Exception as pca_e:
        print(f"Error during PCA visualization: {pca_e}")

    # Save all residue metrics to a file (e.g., CSV for easier analysis)
    try:
        metrics_df = pd.DataFrame(residue_metrics)
        output_path = os.path.join(args.out_dir, "all_residue_metrics.csv")
        metrics_df.to_csv(output_path, index=False)
        print(f"✅ All residue metrics saved to {output_path}")
    except Exception as save_e:
        print(f"Error saving residue metrics to CSV: {save_e}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n✅ Evaluation completed in {end_time - start_time:.2f} seconds.")
