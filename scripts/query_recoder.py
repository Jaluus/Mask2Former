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
    mask2former_transformer_decoder_NOPOS,
)
from mask2former import maskformer_model_CLIP

import torch


class QueryPredictor(DefaultPredictor):
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            (
                predictions,
                mask_pred_results_softmax,
                mask_cls_results_softmax,
            ) = self.model.forward_with_logits([inputs])
            predictions = predictions[0]
            return predictions, mask_pred_results_softmax, mask_cls_results_softmax


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

    eval_type = "val"
    acdc_subset = "rain"
    out_dir = r"ACDC_RESULTS/maskformer2_20Q"

    cfg = setup_cfg(args)

    predictor = QueryPredictor(cfg)

    # save query features
    query_features = (
        predictor.model.sem_seg_head.predictor.query_feat.weight.detach().cpu().numpy()
    )
    np.savetxt(
        f"{out_dir}/queryfeatures_{eval_type}_{acdc_subset}.txt",
        query_features,
        fmt="%s",
    )

    query_predictions = [[] for _ in range(query_features.shape[0])]
    DATASET_PATH = f"datasets/acdc/rgb_anon/{acdc_subset}/{eval_type}"
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

    np.savetxt(
        f"{out_dir}/querypredictions_{eval_type}_{acdc_subset}.txt",
        query_predictions,
        fmt="%s",
    )
