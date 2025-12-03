"""Generate instance masks from metadata.

Creates pixel-level instance masks with color-coded instance IDs.
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.color_mapping import id_to_color
from utils.geometry import draw_rounded_rectangle_mask


def load_metadata(meta_path: Path) -> dict:
    """Load metadata JSON file."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_instance_mask(metadata: dict, screenshot_path: Path) -> np.ndarray:
    """Create instance mask from metadata.
    
    Args:
        metadata: Metadata dictionary with frames
        screenshot_path: Path to screenshot for getting dimensions
        
    Returns:
        Instance mask array (H, W, 3) with RGB colors
    """
    # Load screenshot to get dimensions
    screenshot = cv2.imread(str(screenshot_path))
    if screenshot is None:
        raise ValueError(f"Could not load screenshot: {screenshot_path}")
    
    height, width = screenshot.shape[:2]
    
    # Create mask (background is black)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Skip header and footer containers (they don't have masks)
    # But draw frames that are inside footer (they have masks)
    
    # Note: header and footer are stored separately in metadata['header'] and metadata['footer']
    # They are not in metadata['frames'], so we don't need to skip them here
    
    # Draw each frame (frames inside footer area still get masks)
    for frame in metadata['frames']:
        
        instance_id = frame['id']
        x = frame['x']
        y = frame['y']
        w = frame['w']
        h = frame['h']
        border_radius = frame.get('border_radius', 0)
        
        # Get color for this instance
        color = id_to_color(instance_id)
        
        # Draw rounded rectangle
        # Note: frames inside footer are still drawn (they have masks)
        draw_rounded_rectangle_mask(mask, x, y, w, h, border_radius, color)
    
    return mask


def main():
    parser = argparse.ArgumentParser(description='Generate instance masks from metadata')
    parser.add_argument('--meta-dir', type=str, default='data/meta', help='Directory with metadata JSON files')
    parser.add_argument('--screenshot-dir', type=str, default='data/screenshots', help='Directory with screenshot PNG files')
    parser.add_argument('--output-dir', type=str, default='data/masks', help='Output directory for masks')
    
    args = parser.parse_args()
    
    meta_dir = Path(args.meta_dir)
    screenshot_dir = Path(args.screenshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all metadata files
    meta_files = sorted(meta_dir.glob('page_*.json'))
    
    if not meta_files:
        print(f"No metadata files found in {meta_dir}")
        return
    
    print(f"Found {len(meta_files)} metadata files")
    
    success_count = 0
    for meta_path in tqdm(meta_files, desc="Generating masks"):
        try:
            # Load metadata
            metadata = load_metadata(meta_path)
            page_id = metadata['page_id']
            
            # Find corresponding screenshot
            screenshot_path = screenshot_dir / f"page_{page_id}.png"
            if not screenshot_path.exists():
                print(f"Warning: Screenshot not found: {screenshot_path}")
                continue
            
            # Create mask
            mask = create_instance_mask(metadata, screenshot_path)
            
            # Save mask
            output_path = output_dir / f"page_{page_id}_instance_mask.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            
            success_count += 1
        except Exception as e:
            print(f"Error processing {meta_path}: {e}")
    
    print(f"Done! Generated {success_count}/{len(meta_files)} masks")
    print(f"Masks saved in {output_dir}")


if __name__ == '__main__':
    main()

