import argparse
import json
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

LOREM = [
    "Lorem ipsum dolor sit amet.",
    "Consectetur adipiscing elit.",
    "Vestibulum vitae.",
    "Phasellus placerat enim.",
    "Morbi nec metus.",
    "Nulla facilisi.",
    "Maecenas non leo.",
    "Porttitor mattis sapien."
]
IMAGES = [
    "https://picsum.photos/seed/{}/200/300",
    "https://loremflickr.com/320/240/landscape?lock={}"
]

# Viewport size for converting vw/vh to pixels
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080

#checked
def iou(box1, box2):
    # box = (x1, y1, x2, y2)
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou_value = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou_value

#checked
def random_block_non_overlapping(index, placed_boxes, section_y_offset_vh=0):
    """Generate a random block with metadata.
    
    Returns:
        tuple: (html_string, metadata_dict) or (None, None) if failed
    """
    for attempt in range(100):  # Ограничение на попытки
        w = random.randint(10, 50)
        h = random.randint(10, 50)
        # Генерация координат внутри окна (0-100 vw/vh)
        # Учитываем, чтобы блок не выходил за границы viewport
        max_x = max(0, 100 - w)
        max_y = max(0, 100 - h)
        x = random.randint(0, max_x) if max_x >= 0 else 0
        y = random.randint(0, max_y) if max_y >= 0 else 0
        new_box = (x, y, x + w, y + h) # (x1, y1, x2, y2)
        if sum(iou(new_box, existing) for existing in placed_boxes) < 0.1:
            placed_boxes.append(new_box)
            
            # Convert vw/vh to pixels for metadata
            x_px = int(x * VIEWPORT_WIDTH / 100)
            y_px = int((y + section_y_offset_vh) * VIEWPORT_HEIGHT / 100)
            w_px = int(w * VIEWPORT_WIDTH / 100)
            h_px = int(h * VIEWPORT_HEIGHT / 100)
            
            # Generate metadata
            z_index = random.randint(1, 999)
            border_radius_px = random.choice([0, random.randint(1, min(w_px, h_px) // 4)])
            bg_color = f"rgb({random.randint(200, 255)}, {random.randint(200, 255)}, {random.randint(200, 255)})"
            box_shadow = f"0 {random.randint(2, 8)}px {random.randint(8, 16)}px rgba(0,0,0,0.{random.randint(1, 3)})"
            
            metadata = {
                'id': index + 1,
                'x': x_px,
                'y': y_px,
                'w': w_px,
                'h': h_px,
                'z_index': z_index,
                'border_radius': border_radius_px,
                'bg_color': bg_color,
                'box_shadow': box_shadow,
                'in_header': False,
                'in_footer': False
            }
            
            is_image = random.choice([True, False])
            html = f'<div class="block" style="width:{w}vw; height:{h}vh; position:absolute; top:{y}vh; left:{x}vw; z-index:{z_index}; border-radius:{border_radius_px}px; background:{bg_color}; box-shadow:{box_shadow};">'
            if is_image:
                img_url = random.choice(IMAGES).format(random.randint(1, 10000))
                html += f'<img src="{img_url}" alt="random" style="width:100%; height:60%;">'
            html += f'<p>{random.choice(LOREM)}</p></div>'
            return html, metadata
    # Если после 100 попыток не нашёл место — вернуть None
    return None, None



def generate_full_page(page_id, num_sections, min_blocks_per_section=3, max_blocks_per_section=10):
    """Generate a full HTML page with metadata.
    
    Args:
        page_id: Page ID
        num_sections: Number of sections (frames)
        min_blocks_per_section: Minimum blocks per section
        max_blocks_per_section: Maximum blocks per section
        
    Returns:
        tuple: (html_string, metadata_dict)
    """
    section_names = ['Hero', 'About', 'Features', 'Contact', 'Footer']
    sections = []
    all_frames_metadata = []
    frame_id_counter = 1
    
    for i in range(num_sections):
        name = section_names[i] if i < len(section_names) else f'Section {i+1}'
        placed_boxes = []
        blocks = []
        
        # Random number of blocks per section
        num_blocks = random.randint(min_blocks_per_section, max_blocks_per_section)
        section_y_offset_vh = i * 100  # Each section is 100vh tall
        
        for j in range(num_blocks):
            html, metadata = random_block_non_overlapping(frame_id_counter - 1, placed_boxes, section_y_offset_vh)
            if html:
                blocks.append(html)
                if metadata:
                    metadata['id'] = frame_id_counter
                    all_frames_metadata.append(metadata)
                    frame_id_counter += 1
        
        blocks_html = '\n'.join(blocks)
        section_html = f'<section class="section s{i+1}">{name} секция{blocks_html}</section>'
        sections.append(section_html)
    
    section_styles = '\n'.join([
        f".s{i+1} {{ background: hsl({i*60%360}, 50%, 92%); }}" for i in range(num_sections)
    ])
    
    css = f"""
    <style>
      * {{
        box-sizing: border-box;
      }}
      html, body {{
        margin: 0; 
        padding: 0; 
        width: 100vw;
        height: 100%; 
        overflow-x: hidden;
        box-sizing: border-box; 
        font-family: Arial, sans-serif;
      }}
      .section {{
        width: 100vw; 
        min-height: 100vh; 
        position: relative; 
        z-index: 1;
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: flex-start;
        font-size: 2rem; 
        scroll-snap-align: start; 
        transition: background .3s; 
        padding: 2rem;
        overflow-x: hidden;
      }}
      {section_styles}
      .block {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        padding: 1vw;
        margin: 1rem 0;
        overflow: hidden;
        max-width: 100vw;
      }}
      .block p {{
        margin: 1vw 0 0 0;
        font-size: 1.1rem;
        text-align: center;
        word-wrap: break-word;
      }}
      body {{
        scroll-snap-type: y mandatory;
        overflow-y: scroll;
        overflow-x: hidden;
        height: 100vh;
      }}
    </style>
    """
    
    # Calculate page height (each section is 100vh)
    page_height = num_sections * VIEWPORT_HEIGHT
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Page {page_id}</title>
  {css}
</head>
<body>
  {''.join(sections)}
</body>
</html>
"""
    
    # Create metadata
    metadata = {
        'page_id': page_id,
        'page_width': VIEWPORT_WIDTH,
        'page_height': page_height,
        'frames': all_frames_metadata
    }
    
    return html, metadata


def generate_and_save_page(args_tuple):
    """Generate and save a single page (worker function for multiprocessing).
    
    Args:
        args_tuple: Tuple of (page_id, min_frames, max_frames, min_blocks, max_blocks, pages_dir, meta_dir)
        
    Returns:
        Tuple of (page_id, success, error_message)
    """
    page_id, min_frames, max_frames, min_blocks_per_section, max_blocks_per_section, pages_dir, meta_dir = args_tuple
    
    try:
        # Random number of sections (frames) for this page (generated in worker process)
        num_sections = random.randint(min_frames, max_frames)
        
        # Generate page
        html, metadata = generate_full_page(
            page_id, num_sections, 
            min_blocks_per_section=min_blocks_per_section,
            max_blocks_per_section=max_blocks_per_section
        )
        
        # Save HTML
        html_path = pages_dir / f"page_{page_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Metadata will be generated by playwright_render.py based on actual rendered coordinates
        # Don't save metadata here - it will be generated when rendering screenshots
        
        return (page_id, True, None)
    except Exception as e:
        return (page_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Generate HTML pages with metadata')
    parser.add_argument('--n', type=int, default=100, help='Number of pages to generate')
    parser.add_argument('--min-frames', type=int, default=3, help='Minimum number of sections (frames) per page')
    parser.add_argument('--max-frames', type=int, default=10, help='Maximum number of sections (frames) per page')
    parser.add_argument('--pages-dir', type=str, default='data/pages', help='Directory to save HTML files')
    parser.add_argument('--meta-dir', type=str, default='data/meta', help='Directory to save metadata JSON files')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    pages_dir = Path(args.pages_dir)
    meta_dir = Path(args.meta_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Set number of workers
    if args.num_workers is None:
        num_workers = cpu_count()
    else:
        num_workers = args.num_workers
    
    print(f"Generating {args.n} pages...")
    print(f"Frames per page: {args.min_frames} to {args.max_frames}")
    print(f"Using {num_workers} worker(s)")
    
    # Prepare arguments for each page
    page_args = []
    for page_id in range(1, args.n + 1):
        page_args.append((
            page_id, 
            args.min_frames,  # min_frames (will be used to generate random num_sections in worker)
            args.max_frames,  # max_frames
            3,  # min_blocks_per_section
            10,  # max_blocks_per_section
            pages_dir, 
            meta_dir
        ))
    
    # Process pages in parallel or sequentially
    if num_workers > 1 and args.n > 1:
        # Parallel processing
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_and_save_page, page_args),
                total=len(page_args),
                desc="Generating pages"
            ))
    else:
        # Sequential processing (faster for small datasets or single worker)
        results = [
            generate_and_save_page(args_tuple)
            for args_tuple in tqdm(page_args, desc="Generating pages")
        ]
    
    # Check results
    success_count = sum(1 for _, success, _ in results if success)
    error_count = sum(1 for _, success, _ in results if not success)
    
    if error_count > 0:
        print(f"\nWarnings: {error_count} page(s) failed to generate:")
        for page_id, success, error in results:
            if not success:
                print(f"  Page {page_id}: {error}")
    
    print(f"\nDone! Generated {success_count}/{args.n} pages")
    print(f"HTML files saved in {pages_dir}")
    print(f"Metadata will be generated when rendering screenshots with playwright_render.py")

if __name__ == "__main__":
    main()

