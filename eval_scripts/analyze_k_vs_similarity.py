### analyze_k_vs_similarity.py
"""
Analyzes impact of k_atoms on embedding similarity.
Loops through LMDBs (k8/, k16/, ...), compares each to original embeddings,
computes metrics via compare_core, finds optimal k per metric, and plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from utils import compare_core

# Configuration
base_dir = "/scratch/groups/rbaltman/ziyiw23/clps_embed/k_benchmark_results"
original_dir = "/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res_1000"
k_list = sorted([int(d[1:]) for d in os.listdir(base_dir) if d.startswith("k")])

# Metrics
metric_names = ['cosine', 'pcc', 'l2_diff', 'mse', 'spearman']
k_to_metrics = {}

# Compute metrics for each k_atoms LMDB
for k in k_list:
    opt_dir = os.path.join(base_dir, f"k{k}")
    metrics = compare_core.compute_summary_metrics(original_dir, opt_dir)
    k_to_metrics[k] = metrics
df = pd.DataFrame.from_dict(k_to_metrics, orient='index').sort_index()

# Plot all metrics
plt.figure(figsize=(12, 7))
for metric in metric_names:
    plt.plot(df.index, df[metric], marker='o', label=metric)
plt.xlabel("k_atoms")
plt.ylabel("Metric Value")
plt.title("k_atoms vs. Embedding Similarity Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_dir = "/scratch/groups/rbaltman/ziyiw23/COLLAPSE/"
plt.savefig(os.path.join(out_dir, "k_vs_metrics_summary.png"))
plt.close()

# Elbow detection
optimal_k = {}
for metric in metric_names:
    y = df[metric].values
    direction = 'increasing' if metric in ['cosine', 'flipped_cosine', 'pcc', 'spearman'] else 'decreasing'
    shape = 'concave' if direction == 'increasing' else 'convex'
    try:
        knee = KneeLocator(df.index, y, direction=direction, curve=shape)
        optimal_k[metric] = knee.knee
    except:
        optimal_k[metric] = None

# Print detailed metric values for each k
print("\nEmbedding Similarity Metrics by k_atoms:\n")
for k in df.index:
    print(f"When k = {k}:")
    for metric in metric_names:
        print(f"{metric:>10} = {df.loc[k, metric]:.4f}")
    print("=" * 25)

# Print optimal k_atoms per metric
summary_df = pd.DataFrame(optimal_k, index=["Optimal k_atoms"]).T
print("\nOptimal k_atoms per metric:\n")
print(summary_df)
summary_df.to_csv(os.path.join(out_dir, "optimal_k_per_metric.csv"))
