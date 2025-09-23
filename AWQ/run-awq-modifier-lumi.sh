#!/bin/bash                                                                                                                   
#SBATCH --account=project_xxxxxxxx                                                                                          
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
module use /appl/local/csc/modulefiles                                                                                        
module load pytorch                                                                                                           

# Activate the virtual environment from your current directory or change to the appropriate path
source venv/bin/activate

# This will store all the Hugging Face cache such as downloaded models                                                        
# and datasets in the project's scratch folder                                                                                
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache                                                                 
mkdir -p $HF_HOME                                                                                                             
                                                                                                                              
srun python3 awq-modifier.py  
