#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=compare_embed
#SBATCH --partition=rbaltman
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/compare_embed_output.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/compare_embed_error.log

# Navigate to the working directory
cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate collapse

# Define paths
SCRIPT="eval_scripts/compare_embed.py"
# ORIGINAL_DB="/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res"
ORIGINAL_DB="/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res_1000"
# OPTIMIZED_DB="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res"
OPTIMIZED_DB="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res_1000"
# SECOND_ORI_DB="/scratch/groups/rbaltman/ziyiw23/test_results"
OUT_DIR="/scratch/groups/rbaltman/ziyiw23/COLLAPSE/embed_eval_output/"

# Create OUT_DIR if it doesn't exist
if [ ! -d "$OUT_DIR" ]; then
    echo "Creating output directory: $OUT_DIR"
    mkdir -p "$OUT_DIR"
fi

# Clean up the output directory
rm -rf "$OUT_DIR"/*

echo "Running compare_embed.py..."
python3 "$SCRIPT" --original "$ORIGINAL_DB" --out_dir "$OUT_DIR" \
    --optimized "$OPTIMIZED_DB" \
    --inspect \
    # --second_ori "$SECOND_ORI_DB"

echo "Execution of compare_embed.py completed."
