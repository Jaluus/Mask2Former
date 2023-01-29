#!/bin/bash

model_name=maskformer2_R50_bs16_90k

python train_net.py \
  --config-file ./configs/cityscapes/semantic-segmentation/${model_name}.yaml \
  --num-gpus 8 \
  OUTPUT_DIR ./output_multi/${model_name}