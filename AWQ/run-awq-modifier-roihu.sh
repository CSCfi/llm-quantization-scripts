#!/bin/bash
#SBATCH --account=project_xxxxxxx
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=32G

module purge
module load python-pytorch/2.10

# Activate the virtual environment from your current directory or change to the appropriate path
source venv/bin/activate

# Set hf cache to the project's scratch
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache/
mkdir -p $HF_HOME

srun python3 awq-modifier.py

