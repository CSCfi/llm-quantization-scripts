#!/bin/bash
#SBATCH --argos=no
#SBATCH --account=project_xxxxxxx
#SBATCH --partition=gpumedium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=240G

module purge
module load python-pytorch/2.10

# Activate the virtual environment from your current directory or change to the appropriate path
source bbvenv/bin/activate

# We are putting the cache in the ramdisk, stored in
# memory. Alternatively store it to the project's scratch.
#export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache/
export HF_HOME=/dev/shm/$USER/hf-cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/$USER/
mkdir -p $HF_HOME

srun python3 awq-modifier.py

