"""
Helper utilities for registering the single-class COCO-style dataset.

The module exposes :func:`register_single_class_coco`, which can be imported by
training scripts to ensure Detectron2 knows how to fetch the dataset before
building dataloaders.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def _validate_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _register(
    name: str, annotation_file: Path, image_root: Path, thing_classes: Sequence[str]
) -> None:
    register_coco_instances(name, {}, str(annotation_file), str(image_root))
    metadata = MetadataCatalog.get(name)
    metadata.thing_classes = list(thing_classes)


def register_single_class_coco(
    *,
    dataset_root: str | Path = "datasets/my_coco",
    images_subdir: str | Path = "images",
    train_json: str | Path = "annotations/instances_train.json",
    val_json: str | Path = "annotations/instances_val.json",
    train_name: str = "my_coco_train",
    val_name: str = "my_coco_val",
    thing_classes: Iterable[str] = ("target",),
) -> tuple[str, str]:
    """
    Register COCO-format train/val splits that each contain a single foreground class.

    Args:
        dataset_root: Root directory containing the images/annotations folders.
        images_subdir: Relative directory (inside dataset_root) with all images.
        train_json: Relative path to the training annotation json.
        val_json: Relative path to the validation annotation json.
        train_name: DatasetCatalog name for the training split.
        val_name: DatasetCatalog name for the validation split.
        thing_classes: Iterable of class names; by default a single entry ``"target"``.

    Returns:
        Tuple of the (train_name, val_name) for convenience.
    """

    dataset_root = Path(dataset_root)
    image_root = dataset_root / images_subdir
    train_ann = dataset_root / train_json
    val_ann = dataset_root / val_json

    _validate_path(image_root, "Image directory")
    _validate_path(train_ann, "Training annotation file")
    _validate_path(val_ann, "Validation annotation file")

    thing_classes = tuple(thing_classes)
    if not thing_classes:
        raise ValueError("At least one thing class must be provided.")

    _register(train_name, train_ann, image_root, thing_classes)
    _register(val_name, val_ann, image_root, thing_classes)
    return train_name, val_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register single-class COCO datasets with Detectron2."
    )
    parser.add_argument("--dataset-root", default="datasets/my_coco", type=str)
    parser.add_argument("--images-subdir", default="images", type=str)
    parser.add_argument("--train-json", default="annotations/instances_train.json", type=str)
    parser.add_argument("--val-json", default="annotations/instances_val.json", type=str)
    parser.add_argument("--train-name", default="my_coco_train", type=str)
    parser.add_argument("--val-name", default="my_coco_val", type=str)
    parser.add_argument(
        "--thing-classes",
        nargs="+",
        default=["target"],
        help="Space-separated list of class names. Defaults to a single 'target' class.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_name, val_name = register_single_class_coco(
        dataset_root=args.dataset_root,
        images_subdir=args.images_subdir,
        train_json=args.train_json,
        val_json=args.val_json,
        train_name=args.train_name,
        val_name=args.val_name,
        thing_classes=args.thing_classes,
    )
    print(
        f"Registered datasets '{train_name}' and '{val_name}' "
        f"with classes: {', '.join(args.thing_classes)}"
    )

