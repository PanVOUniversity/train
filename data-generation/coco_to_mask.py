"""Convert COCO annotations back to instance masks.

Reads COCO JSON format and creates color-coded instance masks using pycocotools.
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from operator import itemgetter
from pycocotools import mask as coco_mask
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.color_mapping import id_to_color


def decode_rle_to_mask(rle: dict) -> np.ndarray:
    """Decode RLE segmentation to binary mask using pycocotools.
    
    Args:
        rle: RLE dictionary with 'size' [width, height] (COCO format) and 'counts'
        
    Returns:
        Binary mask (H, W) with 255 for object pixels, 0 for background
    """
    # COCO format uses [width, height], but pycocotools expects [height, width]
    # Convert COCO format [width, height] to pycocotools format [height, width]
    coco_size = rle['size']  # [width, height]
    pycocotools_size = [coco_size[1], coco_size[0]]  # [height, width]
    
    # Prepare RLE for decoding (ensure counts is bytes)
    rle_for_decode = {
        'size': pycocotools_size,
        'counts': rle['counts'].encode('utf-8') if isinstance(rle['counts'], str) else rle['counts']
    }
    
    # Decode RLE to binary mask and convert to uint8
    return (coco_mask.decode(rle_for_decode) * 255).astype(np.uint8)


def coco_to_mask(coco_json_path: Path, output_dir: Path, split: str = 'train'):
    """Convert COCO annotations back to instance masks.
    
    Args:
        coco_json_path: Path to COCO JSON file (e.g., instances_train.json)
        output_dir: Output directory for masks
        split: Dataset split ('train' or 'val')
    """
    # Load COCO JSON
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create image lookup dictionary
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    # Group annotations by image_id using defaultdict
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[ann['image_id']].append(ann)
    
    print(f"Processing {len(images)} images with {len(annotations)} annotations...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for image_id, image_info in tqdm(images.items(), desc=f"Converting {split} to masks"):
        try:
            width = image_info['width']
            height = image_info['height']
            
            # Create instance mask (RGB)
            instance_mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get annotations for this image and sort by z_index
            image_annotations = sorted(
                annotations_by_image[image_id],
                key=itemgetter('z_index'),
                reverse=False  # Lower z_index drawn first (background to foreground)
            )
            
            # Assign instance IDs starting from 1
            instance_id = 1
            
            for ann in image_annotations:
                # Decode RLE to binary mask
                rle = ann['segmentation']
                binary_mask = decode_rle_to_mask(rle)
                
                # Check and resize mask if needed
                # RLE size is [width, height], but decoded mask is (height, width)
                rle_height, rle_width = binary_mask.shape
                if rle_height != height or rle_width != width:
                    # Resize mask to match image dimensions
                    binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    # Threshold to ensure binary values
                    binary_mask = (binary_mask > 127).astype(np.uint8) * 255
                
                # Get color for this instance
                color = id_to_color(instance_id)
                
                # Draw this instance on the mask
                # Only draw where binary_mask is non-zero
                mask_indices = binary_mask > 0
                instance_mask[mask_indices] = color
                
                instance_id += 1
            
            # Save instance mask
            page_id = image_id
            output_path = output_dir / f"page_{page_id}_instance_mask.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(instance_mask, cv2.COLOR_RGB2BGR))
        
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Done! Created instance masks in {output_dir}")


def convert_coco_to_masks(coco_dir: Path, output_dir: Path):
    """Convert COCO dataset back to instance masks.
    
    Args:
        coco_dir: Directory containing COCO dataset (with annotations/ subdirectory)
        output_dir: Output directory for masks
    """
    annotations_dir = coco_dir / 'annotations'
    
    # Process both splits
    for split in ['train', 'val']:
        coco_json = annotations_dir / f"instances_{split}.json"
        if not coco_json.exists():
            print(f"Warning: {coco_json} not found, skipping {split} split")
            continue
        
        split_output_dir = output_dir / split
        print(f"\nConverting {split} split...")
        coco_to_mask(coco_json, split_output_dir, split)


def main():
    parser = argparse.ArgumentParser(description='Convert COCO annotations to instance masks')
    parser.add_argument('--coco-dir', type=str, default='data/coco', 
                        help='Directory containing COCO dataset (with annotations/ subdirectory)')
    parser.add_argument('--output-dir', type=str, default='data/masks_from_coco', 
                        help='Output directory for masks')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val'],
                        help='Specific split to convert (if not provided, converts both)')
    
    args = parser.parse_args()
    
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    
    if args.split:
        # Convert only specified split
        annotations_dir = coco_dir / 'annotations'
        coco_json = annotations_dir / f"instances_{args.split}.json"
        if not coco_json.exists():
            print(f"Error: {coco_json} not found")
            return
        
        split_output_dir = output_dir / args.split
        print(f"Converting {args.split} split...")
        coco_to_mask(coco_json, split_output_dir, args.split)
    else:
        # Convert both splits
        convert_coco_to_masks(coco_dir, output_dir)
    
    print("\nDone! Instance masks created from COCO dataset.")


if __name__ == '__main__':
    main()

