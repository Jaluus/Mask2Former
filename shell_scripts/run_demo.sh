#!/bin/bash

python demo/demo.py --config-file ./configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --input ./demo/img1.png ./demo/img2.png \
  --output ./demo/output_ORIGINAL \
  --opts MODEL.WEIGHTS ./models/model_final_4ab90c.pkl