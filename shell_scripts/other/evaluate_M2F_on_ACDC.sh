#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name="Evaluate M2F on ACDC"
#SBATCH --time=1-00:00:00
#SBATCH --partition=dgx2
#SBATCH --account=supp0003
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --mail-user=jan-lucas.uslu@rwth-aachen.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/ut964798/logs/slurm_logs/%j_%n_%x.txt
#SBATCH --error=/home/ut964798/logs/slurm_logs/%j_%n_%x.err
# format like node+jobname+user+jobid

export OMP_NUM_THREADS=12

# Loading Modules
# first unload all modules
module unload cudnn cuda gcc
module load gcc/9 cuda/11.6 cudnn/8.4.0 

# Activate the conda environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate m2f37

# run the eval
python $HOME/Mask2Former/evaluate_models.py