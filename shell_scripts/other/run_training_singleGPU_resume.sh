#!/bin/bash

model_name=maskformer2_20Q_CLIPBB_CLIPEMB_NOMATCH

python train_net.py \
  --config-file ./configs/cityscapes/semantic-segmentation/${model_name}.yaml \
  --resume \
  --num-gpus 1 \
  MODEL.WEIGHTS ./output/${model_name}/model_interrupted_53.pth \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.MAX_ITER 90000 \
  MODEL.RESNETS.NORM "BN" \
  OUTPUT_DIR ./output/${model_name}