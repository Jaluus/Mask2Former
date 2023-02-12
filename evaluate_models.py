import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import cv2
import os
import numpy as np
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

from mask2former import add_maskformer2_config
from datasets.cityscapes import Cityscapes

from mask2former.modeling.backbone.resnet_CLIP import build_CLIP_backbone
from mask2former.modeling.transformer_decoder import (
    mask2former_transformer_decoder_CLIP,
    mask2former_transformer_decoder_NOPOS,
)
from mask2former import maskformer_model_CLIP


from metrics.stream_metrics import StreamSegMetrics
import torch


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
HPCWORK_DIR = os.environ["HPCWORK"]
WORK_DIR = os.environ["WORK"]

MODEL_DIR = os.path.join(HOME_DIR, "Mask2Former", "CLAIX_OUTPUT")
SAVE_DIR = os.path.join(HOME_DIR, "Mask2Former", "CLAIX_OUTPUT")
MODEL_NAMES = os.listdir(MODEL_DIR)

data_root = os.path.join(WORK_DIR, "datasets", "acdc")
dataset_name = "ACDC"
ACDC_subs = ["night", "rain", "snow"]
split = "val"

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for model_name in MODEL_NAMES:
        CONFIG_FILE = os.path.join(MODEL_DIR, model_name, "config.yaml")
        OPTS = ["MODEL.WEIGHTS", os.path.join(MODEL_DIR, model_name, "model_final.pth")]

        cfg = setup_cfg(config_file=CONFIG_FILE, opts=OPTS)
        predictor = DefaultPredictor(cfg)

        for ACDC_sub in ACDC_subs:
            metrics = StreamSegMetrics(19)
            dataset = Cityscapes(
                root=data_root,
                dataset=dataset_name,
                split=split,
                ACDC_sub=ACDC_sub,
            )

            for idx, (_, _, image, target) in enumerate(dataset):
                print(
                    f"Processing image {idx} of {len(dataset)} in {ACDC_sub} for {model_name}",
                    end="\r",
                )
                image = np.array(image)
                target = np.array(target)

                predictions = predictor(image)
                mask = predictions["sem_seg"].argmax(dim=0).to("cpu").numpy()

                metrics.update(target, mask)

            score = metrics.get_results()

            save_path = os.path.join(
                SAVE_DIR, model_name, f"results_{ACDC_sub}_{split}_{model_name}.json"
            )

            with open(save_path, "w") as f:
                json.dump(score, f, indent=4)
