#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=ppi_clps
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/ppi.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/ppi_error.log

source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/envs/collapse 

SCRIPT="gen_embed.py"
DATA_IN="/scratch/groups/rbaltman/ziyiw23/ProteinWorkshop/data"
DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res"
PROFILE_OUT="profile_opt.prof"
ROOT="/scratch/groups/rbaltman/ziyiw23/ProteinWorkshop/data/rcsb_pdbs"
SPLITS=("train" "val" "test")

# === PROCESS EACH SPLIT ===
for SPLIT in "${SPLITS[@]}"; do
    DATA_IN="${ROOT}/${SPLIT}"
    DATA_OUT="${ROOT}/clps_embed/${SPLIT}"

    echo "Processing ${SPLIT} set..."
    mkdir -p "${DATA_OUT}"

    python3 "$SCRIPT" "$DATA_IN" "$DATA_OUT" --filetype pdb

    echo "Finished ${SPLIT}, results saved to ${DATA_OUT}"
done

echo "All splits processed successfully."
