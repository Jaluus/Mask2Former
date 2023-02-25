#!/bin/bash

#SBATCH --cpus-per-task=12
#SBATCH --mem=200GB
#SBATCH --gres=gpu:8
#SBATCH --job-name="M2F 20Q 1024DIM CLIPBB FROZEN NOPOS CLIPEMB"
#SBATCH --time=2-00:00:00
#SBATCH --partition=dgx2
#SBATCH --account=supp0003
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --mail-user=jan-lucas.uslu@rwth-aachen.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/ut964798/logs/slurm_logs/%j_%n_%x.txt
#SBATCH --error=/home/ut964798/logs/slurm_logs/%j_%n_%x.err
# format like node+jobname+user+jobid

# Set the model name
model_name=maskformer2_20Q_1024DIM_CLIPBB_FROZEN_NOPOS_CLIPEMB

# copy the dataset to the node
# (should improve the performance significantly)
# Currently does nothing, weirdly enough
# The dataloaders are still extremely slow
rsync -rah $WORK/datasets $HPCWORK

# Export the dataset path
# Detectron2 now looks for the dataset in this path
export DETECTRON2_DATASETS=$HPCWORK/datasets
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

# run the training
python $HOME/Mask2Former/train_net.py \
  --num-gpus 8 \
  --config-file $HOME/Mask2Former/configs/cityscapes/semantic-segmentation/${model_name}.yaml \
  --dist-url auto \
  OUTPUT_DIR $HPCWORK/Mask2Former/CLAIX_OUTPUT/${model_name}