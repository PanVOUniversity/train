"""Color mapping utilities for instance masks.

Maps instance IDs to RGB colors and vice versa.
Background is (0, 0, 0), instance IDs start from 1.
"""


def id_to_color(instance_id: int) -> tuple[int, int, int]:
    """Convert instance ID to RGB color.
    
    Args:
        instance_id: Instance ID (1-based)
        
    Returns:
        RGB tuple (R, G, B) with values 0-255
    """
    if instance_id == 0:
        return (0, 0, 0)  # Background
    
    # Map ID to color using bit manipulation
    # id=1 -> (255,0,0), id=2 -> (0,255,0), id=3 -> (0,0,255), id=4 -> (255,255,0), etc.
    r = ((instance_id - 1) & 1) * 255
    g = (((instance_id - 1) >> 1) & 1) * 255
    b = (((instance_id - 1) >> 2) & 1) * 255
    
    # For IDs > 7, use a more complex mapping
    if instance_id > 7:
        # Use a hash-like function to distribute colors
        r = ((instance_id * 17) % 256)
        g = ((instance_id * 31) % 256)
        b = ((instance_id * 47) % 256)
    
    # Ensure not black (background) for ALL IDs >= 1
    if r == 0 and g == 0 and b == 0:
        r = 255  # Use bright red for ID=1 instead of almost-black
    
    return (r, g, b)


def color_to_id(r: int, g: int, b: int) -> int:
    """Convert RGB color to instance ID.
    
    Args:
        r: Red channel (0-255)
        g: Green channel (0-255)
        b: Blue channel (0-255)
        
    Returns:
        Instance ID (0 for background, 1+ for instances)
    """
    if r == 0 and g == 0 and b == 0:
        return 0  # Background
    
    # Try to reverse the simple mapping first
    if r == 255 and g == 0 and b == 0:
        return 1  # ID=1 is now (255,0,0) - bright red
    if r == 0 and g == 255 and b == 0:
        return 2
    if r == 0 and g == 0 and b == 255:
        return 3
    if r == 255 and g == 255 and b == 0:
        return 4
    if r == 255 and g == 0 and b == 255:
        return 5
    if r == 0 and g == 255 and b == 255:
        return 6
    if r == 255 and g == 255 and b == 255:
        return 7
    
    # For complex colors, use a lookup approach
    # This is approximate - exact reverse mapping may not be possible
    # In practice, we'll iterate through possible IDs
    for test_id in range(1, 1000):
        test_color = id_to_color(test_id)
        if abs(test_color[0] - r) < 5 and abs(test_color[1] - g) < 5 and abs(test_color[2] - b) < 5:
            return test_id
    
    # Fallback: use color as hash
    return ((r << 16) | (g << 8) | b) % 1000 + 1

