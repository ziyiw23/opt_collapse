#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --partition=rbaltman
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=install_deps_%j.log
#SBATCH --error=install_deps_%j.err

# Load Conda and activate your env
source /scratch/groups/rbaltman/ziyiw23/conda_envs/miniconda3/etc/profile.d/conda.sh
conda activate collapse

which python

# Auto-confirm prompts in the install script
yes | bash /scratch/groups/rbaltman/ziyiw23/COLLAPSE/install_dependencies.sh

# Install collapse package in editable mode
pip install -e /scratch/groups/rbaltman/ziyiw23/COLLAPSE