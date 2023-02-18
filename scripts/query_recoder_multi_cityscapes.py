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


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


HOME_DIR = os.environ["HOME"]
# HPCWORK_DIR = os.environ["HPCWORK"]
# WORK_DIR = os.environ["WORK"]

MODEL_DIR = os.path.join(HOME_DIR, "Mask2Former", "CLAIX_OUTPUT")
SAVE_DIR = os.path.join(HOME_DIR, "Mask2Former", "ACDC_RESULTS")
MODEL_NAMES = os.listdir(MODEL_DIR)

os.makedirs(SAVE_DIR, exist_ok=True)

split = "val"

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for model_name in MODEL_NAMES:
        CONFIG_FILE = os.path.join(MODEL_DIR, model_name, "config.yaml")
        OPTS = ["MODEL.WEIGHTS", os.path.join(MODEL_DIR, model_name, "model_final.pth")]

        cfg = setup_cfg(config_file=CONFIG_FILE, opts=OPTS)
        predictor = QueryPredictor(cfg)

        query_features = (
            predictor.model.sem_seg_head.predictor.query_feat.weight.detach()
            .cpu()
            .numpy()
        )

        np.savetxt(
            f"{SAVE_DIR}/{model_name}/query_features.txt",
            query_features,
            fmt="%s",
        )

        query_predictions = [[] for _ in range(query_features.shape[0])]
        DATASET_PATH = f"datasets/cityscapes/leftImg8bit/{split}"
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
            f"{SAVE_DIR}/{model_name}/querypredictions_{split}_cityscapes.txt",
            query_predictions,
            fmt="%s",
        )
