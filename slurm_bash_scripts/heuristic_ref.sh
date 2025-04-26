#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=ori_1000
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/ori1000_output.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/ori1000_error.log

# Change to working directory
cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit

# Activate conda environment
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate collapse

# Define variables
SCRIPT="embed_pdb_dataset.py"
DATA_IN="/scratch/groups/rbaltman/ziyiw23/1000_pdbs"
DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res_1000"

# Clean output directory
rm -rf "$DATA_OUT"/*

echo "⏱️ Starting embedding run at $(date)"
start_time=$(date +%s)

python3 "$SCRIPT" "$DATA_IN" "$DATA_OUT" --filetype pdb

end_time=$(date +%s)
echo "✅ Embedding completed at $(date)"
echo "⏱️ Total runtime: $((end_time-start_time)) seconds"
