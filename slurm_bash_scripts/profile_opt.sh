#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=opt_clps
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
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
# DATA_IN="/scratch/groups/rbaltman/ziyiw23/clps_pdbs/"
DATA_IN="/scratch/groups/rbaltman/ziyiw23/1000_pdbs/"
# DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res"
DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/opt_res_1000"
PROFILE_OUT="profile_1000-1.prof"

rm -rf "$DATA_OUT"/*

# --- Resource Monitoring ---
echo "Starting GPU and CPU monitoring..."
LOG_DIR="logs"
mkdir -p $LOG_DIR # Ensure log directory exists

# Monitor GPU usage (nvidia-smi)
#nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total --format=csv -l 1 > $LOG_DIR/gpu_usage_${SLURM_JOB_ID}.csv &
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total --format=csv -l 1 > $LOG_DIR/gpu_usage_${SLURM_JOB_ID}.csv &
NVIDIA_SMI_PID=$!
echo "nvidia-smi monitoring started (PID: $NVIDIA_SMI_PID)"

# Monitor CPU and Memory usage (sar)
#sar -u -r 1 > $LOG_DIR/cpu_mem_usage_${SLURM_JOB_ID}.log &
# --- MODIFIED: Use -P ALL to get per-core CPU data --- #
sar -P ALL -u -r 1 > $LOG_DIR/cpu_mem_usage_${SLURM_JOB_ID}.log &
# --- END MODIFICATION --- #
SAR_PID=$!
echo "sar monitoring started (PID: $SAR_PID)"
# --- End Resource Monitoring ---

# Environment setup (ensure script and data paths are correct)
# ... [rest of script setup] ...

echo "Final NUM_GPUS value: $NUM_GPUS" # DEBUG
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" # DEBUG - RE-ADDED
# echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME" # DEBUG # DDP specific - removed

# Determine the primary GPU from the list for DataParallel
# PRIMARY_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d ',' -f 1) # Only relevant for DP
# echo "Primary GPU for DataParallel (usually cuda:0 equivalent): $PRIMARY_GPU" # Only relevant for DP

# echo "Running $SCRIPT using python3 (DataParallel will use visible GPUs)..." # DP comment
echo "Running $SCRIPT using python3 (Single GPU: cuda:0 equivalent)..." # Single GPU comment
# --- Launch with python3 (DataParallel) --- ## REVERTED ##
# python3 "$SCRIPT" "$DATA_IN" "$DATA_OUT" --filetype pdb --compile_model # Original DP launch
# --- Launch with python3 (Single GPU) --- ## CURRENT ##
# Execute the python script with cProfile
python3 -m cProfile -o $PROFILE_OUT "$SCRIPT" "$DATA_IN" "$DATA_OUT" --filetype pdb --compile_model --num_workers 9
# --- End Launch ---

echo "Script execution complete."

# --- Stop Monitoring ---
echo "Stopping monitoring processes..."
kill $NVIDIA_SMI_PID
kill $SAR_PID
# Allow a moment for processes to terminate cleanly
sleep 2
echo "Monitoring stopped."
# --- End Stop Monitoring ---

echo "Generating profile report..."
python3 -m pstats "$PROFILE_OUT" <<EOF > profile_1000-1.txt
sort tottime
stats 20
sort cumtime
stats 20
EOF
echo "Profile report generated in profile_opt.txt."

echo "All profiling steps complete."