#!/bin/bash

#SBATCH --job-name=k_benchmark
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/k_benchmark.out
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/k_benchmark.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu

cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit

# Load environment
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate collapse

# Paths
DATA_DIR="/scratch/groups/rbaltman/ziyiw23/1000_pdbs"
CHECKPOINT="data/checkpoints/collapse_base.pt"
RESULT_DIR="/scratch/groups/rbaltman/ziyiw23/clps_embed/k_benchmark_results"
K_LIST=(8 16 24 32 40 48 64 80 96 128)

mkdir -p "$RESULT_DIR"

start_time=$(date +%s)

# Run heuristic embedding for each k_atoms
for K in "${K_LIST[@]}"; do
    OUTDIR="$RESULT_DIR/k${K}"
    rm -rf "$OUTDIR"
    echo "▶ Embedding with k_atoms = $K"
    python3 heuristic_embed.py "$DATA_DIR" "$OUTDIR" \
        --checkpoint "$CHECKPOINT" \
        --k_atoms "$K" \
        --filetype pdb
done

echo "✅ All heuristic embeddings complete. Running summary analysis..."

# Run final evaluation and visualization
python3 python3 eval_scripts/analyze_k_vs_similarity.py

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "🕒 Total runtime: $runtime seconds"
