#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=opt_clps
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/OPT_collapse_output.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/OPT_collapse_error.log

cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit

source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/envs/collapse 


# Define variables
SCRIPT="gen_embed.py"
DATA_IN="/scratch/groups/rbaltman/ziyiw23/clps_pdbs/"
# DATA_IN="/scratch/groups/rbaltman/ziyiw23/1000_pdbs/"
DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res"
# DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res_1000"
PROFILE_OUT="profile_1000-1.prof"

rm -rf "$DATA_OUT"/*

# echo "Starting GPU and CPU monitoring..."
# LOG_DIR="logs"
# mkdir -p $LOG_DIR 

# nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total --format=csv -l 1 > $LOG_DIR/gpu_usage.csv &
# NVIDIA_SMI_PID=$!
# echo "nvidia-smi monitoring started (PID: $NVIDIA_SMI_PID)"


# sar -P ALL -u -r 1 > $LOG_DIR/cpu_mem_usage.log &
# SAR_PID=$!
# echo "sar monitoring started (PID: $SAR_PID)"

echo "Final NUM_GPUS value: $NUM_GPUS" 
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" 

echo "Running $SCRIPT using python3 (Single GPU: cuda:0 equivalent)..."
python3 -m cProfile -o $PROFILE_OUT "$SCRIPT" "$DATA_IN" "$DATA_OUT" --filetype pdb --num_workers 9 --compile_model

# echo "Script execution complete."
# echo "Stopping monitoring processes..."
# kill $NVIDIA_SMI_PID
# kill $SAR_PID

# sleep 2
# echo "Monitoring stopped."

# echo "Generating profile report..."
# python3 -m pstats "$PROFILE_OUT" <<EOF > profile_1000-1.txt
# sort tottime
# stats 20
# sort cumtime
# stats 20
# EOF
# echo "Profile report generated in profile_opt.txt."

echo "All profiling steps complete."