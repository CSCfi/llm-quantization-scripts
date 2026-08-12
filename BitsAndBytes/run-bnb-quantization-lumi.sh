#!/bin/bash
#SBATCH --account=project_xxxxxxx
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load the module
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# export path to used container image
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif

# This will store all the Hugging Face cache such as downloaded models
# and datasets in the project's scratch folder
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/llm-quantization-scripts/BitsAndBytes/hf-cache
mkdir -p $HF_HOME
export SINGULARITYENV_HF_HOME=$HF_HOME

srun singularity exec "$SIF" bash -c 'python3 bnb-quantization.py'
