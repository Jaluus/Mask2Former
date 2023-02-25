#!/bin/bash

model_name=20Q_CLIP_NOPOS_NOMATCH

python train_net.py \
  --config-file ./output_${model_name}/config.yaml \
  --num-gpus 1 \
  --eval-only \
  MODEL.WEIGHTS ./output_${model_name}/model_final.pth \