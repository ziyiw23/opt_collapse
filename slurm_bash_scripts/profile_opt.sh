#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=opt_clps
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/OPT_collapse_output.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/OPT_collapse_error.log

# Change to working directory
cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit

# (while true; do nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits >> gpu_memory.log; sleep 0.5; done) &
# LOG_PID=$!

# Load modules, activate conda environment
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/envs/collapse 


# Define variables
SCRIPT="gen_embed.py"
DATA_IN="/scratch/groups/rbaltman/ziyiw23/clps_pdbs/"
# DATA_IN="/scratch/groups/rbaltman/ziyiw23/1000_pdbs/"
DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res"
# DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res_1000"
PROFILE_OUT="profile_opt.prof"

rm -rf "$DATA_OUT"/*

echo "Running cProfile on $SCRIPT..."
python3 -m cProfile -o "$PROFILE_OUT" "$SCRIPT" "$DATA_IN" "$DATA_OUT" --filetype pdb --compile_model
echo "cProfile complete. Profile data saved in $PROFILE_OUT."

# kill $LOG_PID
# echo "GPU memory log saved to gpu_memory.log"

echo "Generating profile report..."
python3 -m pstats "$PROFILE_OUT" <<EOF > profile_opt.txt
sort tottime
stats 20
sort cumtime
stats 20
EOF
echo "Profile report generated in profile_opt.txt."

echo "All profiling steps complete."