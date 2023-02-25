#!/bin/bash

export DETECTRON2_DATASETS=$WORK/datasets

model_name=maskformer2_R50_bs16_90k

module unload cuda
module load cuda/11.6

module unload gcc
module load gcc/7

# Activate the conda environment
source $HOME/miniconda/bin/activate $HOME/miniconda/envs/m2f

# run teh training
python $HOME/Mask2Former/train_net.py \
  --config-file $HOME/Mask2Former/configs/cityscapes/semantic-segmentation/${model_name}.yaml \
  --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.0001 \
  MODEL.RESNETS.NORM "BN" \
  OUTPUT_DIR $HOME/Mask2Former/output_multi/${model_name}