# Single-Class COCO Training Quickstart

This folder contains the helper scripts needed to train the fastest built-in Detectron2 instance-segmentation model (Mask R-CNN R50-FPN 1× schedule) on the custom COCO-style dataset with a single RLE-encoded class.

## 1. Register the Dataset

Ensure the dataset lives under `datasets/my_coco/` (or pass a custom path). The directory should contain:

```
datasets/my_coco/
  images/                      # PNG/JPG files referenced in the annotations
  annotations/
    instances_train.json       # COCO-format annotations with RLE `segmentation`
    instances_val.json
```

Before training, register the splits with Detectron2:

```bash
python train/register_dataset.py \
  --dataset-root datasets/my_coco \
  --images-subdir images \
  --train-json annotations/instances_train.json \
  --val-json annotations/instances_val.json \
  --thing-classes target
```

The script wires the dataset names `my_coco_train` / `my_coco_val` (override with `--train-name` / `--val-name`) and assigns the single class name `target`. Internally it uses `register_coco_instances`, so the resulting loaders fully support RLE masks when `cfg.INPUT.MASK_FORMAT = "bitmask"`.

## 2. Launch Training

`train/train_single_class.py` is a thin wrapper around Detectron2’s `DefaultTrainer` that picks the lightweight Mask R-CNN R50-FPN 1× config and exposes flags tuned for a single class. Example command:

```bash
python train/train_single_class.py \
  --dataset-root datasets/my_coco \
  --output-dir output/my_target \
  --ims-per-batch 4 \
  --max-iter 8000 \
  --lr-steps 6000 7500 \
  --num-gpus 1
```

Key defaults:
- Config file: `configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml`
- Pretrained weights: `mask_rcnn_R_50_FPN_1x` checkpoint from the Detectron2 model zoo
- `MODEL.ROI_HEADS.NUM_CLASSES = 1`
- `INPUT.MASK_FORMAT = "bitmask"` for COCO-style RLE segmentation

Override `--thing-classes`, `--train-dataset`, or `--val-dataset` if you pick different names. All solver hyperparameters (`--base-lr`, `--max-iter`, `--lr-steps`, `--checkpoint-period`, `--eval-period`) are exposed so you can scale training time with dataset size.

## 3. Verification & Monitoring

- Sanity check annotations with `tools/visualize_data.py` or Detectron2’s demo notebook to confirm masks/classes are loaded correctly.
- Training/evaluation logs and metrics land in `OUTPUT_DIR` (default `output/my_target`). Use TensorBoard or the built-in JSON stats for monitoring.
- To run validation only: add `--eval-only --resume` to the previous command and point `--weights` to the checkpoint you want to evaluate.

Following these steps will run the fastest Mask R-CNN variant that still supports instance masks, keeping training time minimal while satisfying the single-class COCO segmentation requirement.

