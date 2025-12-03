"""
Launch Mask R-CNN training on the single-class COCO dataset generated in this project.

Example:
    python train/train_single_class.py \
        --dataset-root datasets/my_coco \
        --output-dir output/my_target \
        --num-gpus 1
"""

from __future__ import annotations

import argparse
import os
from typing import List

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

from train.register_dataset import register_single_class_coco

DEFAULT_CONFIG = "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
DEFAULT_WEIGHTS = (
    "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        evaluators = [COCOEvaluator(dataset_name, output_dir=output_folder)]
        return DatasetEvaluators(evaluators)


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TRAIN = (args.train_dataset,)
    cfg.DATASETS.TEST = (args.val_dataset,)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = args.filter_empty

    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters
    cfg.SOLVER.STEPS = tuple(args.lr_steps)
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def _register_datasets(args: argparse.Namespace) -> None:
    register_single_class_coco(
        dataset_root=args.dataset_root,
        images_subdir=args.images_subdir,
        train_json=args.train_json,
        val_json=args.val_json,
        train_name=args.train_dataset,
        val_name=args.val_dataset,
        thing_classes=args.thing_classes,
    )


def setup(args: argparse.Namespace):
    cfg = setup_cfg(args)
    default_setup(cfg, args)
    return cfg


def main(args: argparse.Namespace):
    _register_datasets(args)
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, model)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def extend_parser(parser):
    parser.set_defaults(config_file=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", default="datasets/my_coco", type=str)
    parser.add_argument("--images-subdir", default="images", type=str)
    parser.add_argument("--train-json", default="annotations/instances_train.json", type=str)
    parser.add_argument("--val-json", default="annotations/instances_val.json", type=str)
    parser.add_argument("--train-dataset", default="my_coco_train", type=str)
    parser.add_argument("--val-dataset", default="my_coco_val", type=str)
    parser.add_argument("--thing-classes", nargs="+", default=["target"], type=str)
    parser.add_argument("--class-name", default="target", type=str, help="Deprecated alias.")

    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--ims-per-batch", default=4, type=int)
    parser.add_argument("--base-lr", default=0.00025, type=float)
    parser.add_argument("--max-iter", default=5000, type=int)
    parser.add_argument("--lr-steps", nargs="+", default=[3500, 4500], type=int)
    parser.add_argument("--warmup-iters", default=1000, type=int)
    parser.add_argument("--checkpoint-period", default=1000, type=int)
    parser.add_argument("--eval-period", default=500, type=int)
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        type=str,
        help="Detectron2 checkpoint to bootstrap the model.",
    )
    parser.add_argument("--output-dir", default="output/my_target", type=str)
    parser.add_argument(
        "--filter-empty",
        action="store_false",
        dest="filter_empty",
        help="Keep empty images when set (default is to filter them out).",
    )
    parser.set_defaults(filter_empty=True)
    return parser


if __name__ == "__main__":
    parser = extend_parser(default_argument_parser())
    args = parser.parse_args()

    # Keep backward compatibility if someone passes a single class via --class-name
    if args.thing_classes == parser.get_default("thing_classes") and args.class_name:
        args.thing_classes = [args.class_name]

    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, args=(args,))

