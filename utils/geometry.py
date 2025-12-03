"""Geometry utilities for frame segmentation.

Functions for computing overlaps, shifts, and drawing rounded rectangles.
"""

import numpy as np
import cv2
from typing import Tuple


def min_shift_to_resolve_overlap(bbox1: Tuple[float, float, float, float], 
                                  bbox2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Compute minimum shift to resolve overlap between two bounding boxes.
    
    Args:
        bbox1: (x, y, w, h) of first bounding box
        bbox2: (x, y, w, h) of second bounding box
        
    Returns:
        Tuple of (dx_top, dy_top, dx_bottom, dy_bottom) where:
        - dx_top, dy_top: shift for bbox1 to resolve overlap
        - dx_bottom, dy_bottom: shift for bbox2 to resolve overlap
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Compute intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right <= x_left or y_bottom <= y_top:
        # No overlap
        return (0.0, 0.0, 0.0, 0.0)
    
    overlap_w = x_right - x_left
    overlap_h = y_bottom - y_top
    
    # Compute shifts for bbox1 (top)
    # Option 1: shift right
    dx_right = overlap_w
    # Option 2: shift left
    dx_left = -overlap_w
    # Option 3: shift down
    dy_down = overlap_h
    # Option 4: shift up
    dy_up = -overlap_h
    
    # Choose minimum magnitude shift
    shifts_top = [
        (dx_right, 0.0),
        (dx_left, 0.0),
        (0.0, dy_down),
        (0.0, dy_up),
    ]
    
    # Compute shifts for bbox2 (bottom)
    shifts_bottom = [
        (-dx_right, 0.0),
        (-dx_left, 0.0),
        (0.0, -dy_down),
        (0.0, -dy_up),
    ]
    
    # Find minimum magnitude shift
    min_mag_top = float('inf')
    min_mag_bottom = float('inf')
    best_top = (0.0, 0.0)
    best_bottom = (0.0, 0.0)
    
    for dx, dy in shifts_top:
        mag = np.sqrt(dx*dx + dy*dy)
        if mag < min_mag_top:
            min_mag_top = mag
            best_top = (dx, dy)
    
    for dx, dy in shifts_bottom:
        mag = np.sqrt(dx*dx + dy*dy)
        if mag < min_mag_bottom:
            min_mag_bottom = mag
            best_bottom = (dx, dy)
    
    return (best_top[0], best_top[1], best_bottom[0], best_bottom[1])


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Extract bounding box from binary mask.
    
    Args:
        mask: Binary mask (H, W) or (H, W, 3) with instance color
        
    Returns:
        (x, y, w, h) bounding box
    """
    if len(mask.shape) == 3:
        # Convert color mask to binary
        mask = (mask.sum(axis=2) > 0).astype(np.uint8)
    
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return (0, 0, 0, 0)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get minimum bounding box containing mask.
    
    Alias for bbox_from_mask for consistency.
    """
    return bbox_from_mask(mask)


def compute_overlap_pixels(mask1: np.ndarray, mask2: np.ndarray) -> int:
    """Compute number of overlapping pixels between two masks.
    
    Args:
        mask1: First mask (binary or color)
        mask2: Second mask (binary or color)
        
    Returns:
        Number of overlapping pixels
    """
    if len(mask1.shape) == 3:
        mask1 = (mask1.sum(axis=2) > 0).astype(np.uint8)
    if len(mask2.shape) == 3:
        mask2 = (mask2.sum(axis=2) > 0).astype(np.uint8)
    
    overlap = np.logical_and(mask1 > 0, mask2 > 0)
    return int(np.sum(overlap))


def draw_rounded_rectangle_mask(image: np.ndarray, x: int, y: int, w: int, h: int, 
                                border_radius: int, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw a rounded rectangle on the mask image.
    
    Args:
        image: Image array (H, W, 3) to draw on
        x: Left coordinate
        y: Top coordinate
        w: Width
        h: Height
        border_radius: Radius of rounded corners
        color: RGB color tuple
        
    Returns:
        Modified image array
    """
    if border_radius <= 0:
        # Simple rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        return image
    
    # Clamp border radius to half of smallest dimension
    border_radius = min(border_radius, min(w, h) // 2)
    
    # Draw rounded rectangle by drawing:
    # 1. Central rectangle
    # 2. Four corner circles/ellipses
    
    # Central rectangle (without corners)
    cv2.rectangle(image, (x + border_radius, y), (x + w - border_radius, y + h), color, -1)
    cv2.rectangle(image, (x, y + border_radius), (x + w, y + h - border_radius), color, -1)
    
    # Four corner circles
    # Top-left
    cv2.ellipse(image, (x + border_radius, y + border_radius), 
                (border_radius, border_radius), 180, 0, 90, color, -1)
    # Top-right
    cv2.ellipse(image, (x + w - border_radius, y + border_radius), 
                (border_radius, border_radius), 270, 0, 90, color, -1)
    # Bottom-left
    cv2.ellipse(image, (x + border_radius, y + h - border_radius), 
                (border_radius, border_radius), 90, 0, 90, color, -1)
    # Bottom-right
    cv2.ellipse(image, (x + w - border_radius, y + h - border_radius), 
                (border_radius, border_radius), 0, 0, 90, color, -1)
    
    return image

