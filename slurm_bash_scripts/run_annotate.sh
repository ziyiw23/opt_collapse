#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=temp_annotate_clps
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/annotation_%j.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/annotation_%j_error.log

# Usage: sbatch this_script.sh [original|optimized]
MODE=${1:-original}

# Validate mode
if [[ "$MODE" != "original" && "$MODE" != "optimized" ]]; then
  echo "Error: Invalid mode '$MODE'. Use 'original' or 'optimized'."
  exit 1
fi

# Update job name to include mode (visible in squeue)
if command -v scontrol &> /dev/null; then
  scontrol update jobid=$SLURM_JOB_ID name="${MODE}_annotate_clps"
fi

# Rest of the script remains unchanged
cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate collapse

# Define paths
SCRIPT="annotate_pdb.py"
CHECKPOINT="data/checkpoints/collapse_base.pt"  
REFERENCE_DB="data/datasets/full_site_db_stats.pkl"
ANNOTATION_OUT="/scratch/groups/rbaltman/ziyiw23/annotation_results/$MODE"
PDB_DIR="/scratch/groups/rbaltman/ziyiw23/clps_pdbs" 

if [ ! -f "$REFERENCE_DB" ]; then
  echo "Error: Database file $REFERENCE_DB not found!"
  exit 1
fi

if ! python3 -c "import pickle; f=open('$REFERENCE_DB','rb'); pickle.load(f)"; then
  echo "Error: $REFERENCE_DB is corrupted. Re-download it!"
  exit 1
fi

rm -rf "$ANNOTATION_OUT"/*
mkdir -p "$ANNOTATION_OUT"

echo "======= Starting $MODE annotation ======="
echo " - Checkpoint: $CHECKPOINT"
echo " - Reference DB: $REFERENCE_DB"
echo " - Output dir: $ANNOTATION_OUT"
echo " - PDB directory: $PDB_DIR"
echo " - Output log: $OUTPUT_LOG"
echo " - Error log: $ERROR_LOG"

# Process all PDB files in the directory
for PDB_FILE in "$PDB_DIR"/*.pdb; do
  if [ ! -f "$PDB_FILE" ]; then
    echo "Warning: No PDB files found in $PDB_DIR"
    break
  fi

  python3 "$SCRIPT" \
    "$PDB_FILE" \
    --mode "$MODE" \
    --db "$REFERENCE_DB" \
    --checkpoint "$CHECKPOINT" \
    --debug
  echo " "
done

echo "======= $MODE annotation completed ======="