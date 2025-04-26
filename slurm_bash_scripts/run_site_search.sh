#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=opt_clps
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/site_search_output.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/site_search_error.log

cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate collapse

# ----------- Configurable Paths ----------------
CHECKPOINT="data/checkpoints/collapse_base.pt"
DB_PICKLE="/home/ziyiwang/mount_folder/data/datasets/full_site_db_stats.pkl"

ORI_EMB="/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res"
OPT_EMB="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res"

ORI_OUT="/scratch/groups/rbaltman/ziyiw23/site_search/site_search_ori.pkl"
OPT_OUT="/scratch/groups/rbaltman/ziyiw23/site_search/site_search_opt.pkl"
# ------------------------------------------------

echo "🔎 Running site search on original embeddings..."
python search_site.py \
  --query_embed $ORI_EMB \
  --db_pickle $DB_PICKLE \
  --checkpoint $CHECKPOINT \
  --out_file $ORI_OUT

echo "✅ Original search complete."

echo "🚀 Running site search on optimized embeddings..."
python search_site.py \
  --query_embed $OPT_EMB \
  --db_pickle $DB_PICKLE \
  --checkpoint $CHECKPOINT \
  --out_file $OPT_OUT

echo "✅ Optimized search complete."

# Evaluate both outputs using a custom script
echo "📊 Comparing results..."