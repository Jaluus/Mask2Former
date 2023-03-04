# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass


# ignore warnings
import warnings
import sys
import torch
from detectron2.modeling import build_model

# MaskFormer
sys.path.append("/home/ut964798/Mask2Former")
from mask2former import (
    maskformer_model_CLIP,
)
from mask2former.modeling.backbone.resnet_CLIP import build_CLIP_backbone
from mask2former.modeling.pixel_decoder import msdeformattn_FROZEN
from mask2former.modeling.transformer_decoder import (
    mask2former_transformer_decoder_CLIP,
    mask2former_transformer_decoder_CLIP_INC,
    mask2former_transformer_decoder_CLIP_INC_GAUSS,
    mask2former_transformer_decoder_CLIP_TEXTPERTURB,
    mask2former_transformer_decoder_NOPOS,
)
from detectron2.checkpoint import DetectionCheckpointer

import os
from detectron2.config import get_cfg
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
import numpy as np


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


REF_MODEL_DIR = "/home/ut964798/Mask2Former/CLAIX_OUTPUT/Cityscapes"
REF_MODEL_NAME = "maskformer2_R50_bs16_90k"
COMPARE_MODEL_DIR = "/home/ut964798/Mask2Former/CLAIX_OUTPUT/ACDC_NIGHT"
COMPARED_NAMES = [
    "maskformer2_R50_bs16_90k",
    "maskformer2_R50_bs16_90k_BB",
    "maskformer2_R50_bs16_90k_BBPD",
    "maskformer2_R50_bs16_90k_BBPDTD",
]

CONFIG_FILE_REF = os.path.join(REF_MODEL_DIR, REF_MODEL_NAME, "config.yaml")
OPTS_REF = [
    "MODEL.WEIGHTS",
    os.path.join(REF_MODEL_DIR, REF_MODEL_NAME, "model_final.pth"),
]

cfg_ref = setup_cfg(config_file=CONFIG_FILE_REF, opts=OPTS_REF)
model_ref = build_model(cfg_ref)
DetectionCheckpointer(model_ref).load(cfg_ref.MODEL.WEIGHTS)
model_ref.eval()

BB_ref = model_ref.backbone
PD_ref = model_ref.sem_seg_head.pixel_decoder
TD_ref = model_ref.sem_seg_head.predictor

BB_REF_PARAMS = list(BB_ref.parameters())
PD_REF_PARAMS = list(PD_ref.parameters())
TD_REF_PARAMS = list(TD_ref.parameters())

print("REF MODEL")


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            continue
        else:
            if "norm" in key_item_1[0]:
                continue
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
                # pass
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")
    return models_differ


for compared_name in COMPARED_NAMES:
    print("Comparing to", compared_name)
    CONFIG_FILE = os.path.join(COMPARE_MODEL_DIR, compared_name, "config.yaml")
    opts = [
        "MODEL.WEIGHTS",
        os.path.join(COMPARE_MODEL_DIR, compared_name, "model_final.pth"),
    ]

    cfg = setup_cfg(config_file=CONFIG_FILE, opts=opts)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # if np.array_equal(BB_PARAMS, BB_REF_PARAMS):
    #     print("Backbone parameters match!")
    # else:
    #     print("Backbone parameters do not match!")

    # if np.array_equal(TD_PARAMS, TD_REF_PARAMS):
    #     print("Transformer decoder parameters match!")
    # else:
    #     print("Transformer decoder parameters do not match!")

    # if np.array_equal(PD_PARAMS, PD_REF_PARAMS):
    #     print("Pixel decoder parameters match!")
    # else:
    #     print("Pixel decoder parameters do not match!")

    diff = compare_models(model_ref, model)
    print("Models differ in", diff, "parameters")
    print("-" * 20)
