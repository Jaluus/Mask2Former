# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

from mask2former import add_maskformer2_config

from mask2former.modeling.backbone.resnet_CLIP import build_CLIP_backbone
from mask2former.modeling.transformer_decoder import (
    mask2former_transformer_decoder_CLIP,
)

import torch


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    eval_type = "train"
    out_dir = "output_panoptic"

    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)

    query_features = (
        predictor.model.sem_seg_head.predictor.query_feat.weight.detach().cpu().numpy()
    )

    np.save(f"{out_dir}/query_features_{eval_type}.npy", query_features)
    np.savetxt(f"{out_dir}/query_features_{eval_type}.txt", query_features, fmt="%s")

    query_predictions = [[] for _ in range(query_features.shape[0])]
    DATASET_PATH = f"datasets/cityscapes/leftImg8bit/{eval_type}"
    cities = os.listdir(DATASET_PATH)
    for c_idx, city in enumerate(cities):
        image_paths = glob.glob(os.path.join(DATASET_PATH, city, "*.png"))
        for i_idx, image_path in enumerate(image_paths):
            print(
                f"Processing {i_idx + 1}/{len(image_paths)} image in {c_idx + 1}/{len(cities)} city",
                end="\r",
            )
            image = read_image(image_path, format="BGR")
            predictions, mask_preds, class_preds = predictor(image)

            bs, num_queries, _, _ = mask_preds.shape

            # run this for every query
            for i in range(num_queries):
                mask = mask_preds[0, i].detach().cpu().numpy()
                q_class = class_preds[0, i].detach().cpu().numpy()
                max_q_class_idx = np.argmax(q_class)

                query_predictions[i].append(max_q_class_idx)

    np.save(f"{out_dir}/query_predictions_{eval_type}.npy", query_predictions)
    np.savetxt(
        f"{out_dir}/query_predictions_{eval_type}.txt", query_predictions, fmt="%s"
    )
