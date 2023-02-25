#!/bin/bash
model_name=maskformer2_R50_bs16_90k

# Activate the conda environment
source $HOME/miniconda/bin/activate $HOME/miniconda/envs/m2f

python $HOME/Mask2Former/train_net.py \
  --config-file $HOME/Mask2Former/configs/cityscapes/semantic-segmentation/${model_name}.yaml \
  --num-gpus 4 \
  OUTPUT_DIR $HOME/Mask2Former/output_multi/${model_name}