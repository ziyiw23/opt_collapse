#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyiw23@stanford.edu
#SBATCH --job-name=ori_clps
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/OP_collapse_output.log
#SBATCH --error=/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/OP_collapse_error.log

cd /scratch/groups/rbaltman/ziyiw23/COLLAPSE || exit

# source /oak/stanford/groups/rbaltman/aderry/miniconda3/etc/profile.d/conda.sh
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
ml devel cuda/11.7.1 gcc/12.4.0
conda activate collapse

SCRIPT="embed_pdb_dataset.py"
# DATA_IN="/scratch/groups/rbaltman/ziyiw23/clps_pdbs/"
DATA_IN="/scratch/groups/rbaltman/ziyiw23/1000_pdbs/"
# DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res"
DATA_OUT="/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res_1000"
PROFILE_OUT="profile_ori.prof"

rm -rf "$DATA_OUT"/*

echo "Running cProfile on $SCRIPT..."
python3 -m cProfile -o "$PROFILE_OUT" "$SCRIPT" "$DATA_IN" "$DATA_OUT" --filetype pdb
echo "cProfile complete. Profile data saved in $PROFILE_OUT."

echo "Generating profile report..."
python3 -m pstats "$PROFILE_OUT" <<EOF > profile_pro.txt
sort tottime
stats 20
sort cumtime
stats 20
EOF
echo "Profile report generated in profile_ori.txt."

echo "All profiling steps complete."


# if using srun
# srun --job-name=collapse_test_run \
#      --partition=rbaltman \
#      --gres=gpu:1 \
#      --mem=128G \
#      --tmp=50G \
#      --time=24:00:00 \
#      --output=TP_collapse_output.log \
#      --error=TP_collapse_error.log \
#      python3 -m cProfile -o profile_test.prof test_embed.py ./profile_dataset/pdb_files ./data/out_embeddings --filetype pdb
