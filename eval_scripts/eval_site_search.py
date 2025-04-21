import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr

df_ori = pd.read_csv("$OUT_ORI")
df_opt = pd.read_csv("$OUT_OPT")

df_ori = df_ori.dropna(subset=['PDB', 'COSINE'])
df_opt = df_opt.dropna(subset=['PDB', 'COSINE'])

df_ori_sorted = df_ori.sort_values('COSINE', ascending=False).reset_index(drop=True)
df_opt_sorted = df_opt.sort_values('COSINE', ascending=False).reset_index(drop=True)

K = 10
top_ori = set(df_ori_sorted['PDB'].head(K))
top_opt = set(df_opt_sorted['PDB'].head(K))
topk_overlap = len(top_ori & top_opt) / K

set_ori = set(df_ori['PDB'])
set_opt = set(df_opt['PDB'])
jaccard = len(set_ori & set_opt) / len(set_ori | set_opt)

shared = list(set_ori & set_opt)
if shared:
    ori_ranks = [df_ori_sorted[df_ori_sorted['PDB'] == pdb].index[0] for pdb in shared]
    opt_ranks = [df_opt_sorted[df_opt_sorted['PDB'] == pdb].index[0] for pdb in shared]
    spearman, _ = spearmanr(ori_ranks, opt_ranks)
    kendall, _ = kendalltau(ori_ranks, opt_ranks)

    ori_scores = [df_ori[df_ori['PDB'] == pdb]['COSINE'].values[0] for pdb in shared]
    opt_scores = [df_opt[df_opt['PDB'] == pdb]['COSINE'].values[0] for pdb in shared]
    pearson, _ = pearsonr(ori_scores, opt_scores)
else:
    spearman = kendall = pearson = float('nan')

print("📊 Comparison Metrics")
print("--------------------------")
print(f"🔟 Top-{K} Overlap:        {topk_overlap:.2f}")
print(f"🧬 Jaccard Similarity:     {jaccard:.2f}")
print(f"📈 Spearman Rank Corr.:    {spearman:.2f}")
print(f"📊 Kendall's Tau:          {kendall:.2f}")
print(f"📉 Cosine Score Pearson:   {pearson:.2f}")