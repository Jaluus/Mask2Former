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

import atexit
import logging
import os

# ignore warnings
import warnings

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from datasets import register_ACDC
from datasets.ACDC_evaluation import ACDCSemSegEvaluator

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
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
from utils.arg_parser import my_argument_parser
from utils.model_trainer import Trainer, killHook

warnings.simplefilter("ignore", UserWarning)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former"
    )
    return cfg


def main(args):
    logger = logging.getLogger("mask2former")

    frozen_transformer_layers = args.frozen_transformer_layers
    if frozen_transformer_layers != "":
        frozen_transformer_layers = [
            int(i) for i in frozen_transformer_layers.split(",")
        ]
    else:
        frozen_transformer_layers = []
    del args.frozen_transformer_layers
    logger.info("Frozen transformer layers: {}".format(frozen_transformer_layers))

    freeze_everything_except_output_FFN = args.freeze_everything_except_output_FFN
    del args.freeze_everything_except_output_FFN

    freeze_backbone = args.freeze_backbone
    del args.freeze_backbone

    freeze_pixel_decoder = args.freeze_pixel_decoder
    del args.freeze_pixel_decoder

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(
        cfg,
        frozen_transformer_layers,
        freeze_everything_except_output_FFN,
        freeze_backbone,
        freeze_pixel_decoder,
    )
    trainer.register_hooks([killHook()])

    def save_model(curr_trainer):
        checkpointer = DetectionCheckpointer(
            curr_trainer.model,
            cfg.OUTPUT_DIR,
            trainer=curr_trainer,
        )
        checkpointer.save(
            f"model_interrupted_{curr_trainer.iter}", iteration=curr_trainer.iter
        )

    atexit.register(save_model, trainer)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = my_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
