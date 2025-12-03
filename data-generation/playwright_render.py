"""Playwright renderer for taking full-page screenshots and generating metadata.

This file renders HTML files and generates metadata based on actual rendered coordinates.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
import argparse
import random
import json
from multiprocessing import cpu_count
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Render HTML files to screenshots using Playwright.")
    parser.add_argument("--input-dir", type=str, default="data/pages", help="Directory containing input HTML files")
    parser.add_argument("--output-dir", type=str, default="data/screenshots", help="Directory for output screenshots")
    parser.add_argument("--meta-dir", type=str, default="data/meta", help="Directory with metadata JSON files")
    parser.add_argument("--workers", type=int, default=None, help="Number of concurrent workers (default: CPU count)")
    parser.add_argument("--viewport-width", type=int, default=1920, help="Viewport width in pixels (default: 1920)")
    parser.add_argument("--viewport-height", type=int, default=1080, help="Viewport height in pixels (default: 1080)")
    parser.add_argument("--mode", type=str, default="random", choices=["phone", "desktop", "random"], 
                       help="Resolution mode: phone, desktop, or random (default: random)")
    return parser.parse_args()


def random_resolution(mode: str = "random"):
    """Get a random resolution based on mode.
    
    Args:
        mode: 'phone', 'desktop', or 'random' (default: 'random')
        
    Returns:
        Tuple of (width, height) in pixels
    """
    # Phone resolutions with weights (percentages)
    phone_resolutions = [
        (390, 844, 9.83),
        (414, 896, 7.86),
        (384, 832, 6.21),
        (360, 800, 6.0),
        (393, 873, 5.78),
        (393, 852, 5.58),
    ]
    
    # Desktop resolutions with weights (percentages)
    desktop_resolutions = [
        (1920, 1080, 31.48),
        (1536, 864, 11.19),
        (2560, 1440, 5.23),
        (1366, 768, 5.21),
        (800, 600, 5.17),
        (1280, 720, 4.33),
    ]
    
    # If random, first choose between phone and desktop
    if mode == "random":
        mode = random.choice(["phone", "desktop"])
    
    # Select resolution based on mode
    if mode == "phone":
        resolutions = phone_resolutions
    elif mode == "desktop":
        resolutions = desktop_resolutions
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'phone', 'desktop', or 'random'")
    
    # Extract resolutions and weights
    res_list = [(w, h) for w, h, _ in resolutions]
    weights = [prob for _, _, prob in resolutions]
    
    # Select resolution based on weights
    selected = random.choices(res_list, weights=weights, k=1)[0]
    
    return selected


async def extract_frames_metadata(page) -> list:
    """Extract metadata for all frame elements (blocks) from the rendered page.
    
    Args:
        page: Playwright page object
        
    Returns:
        List of frame metadata dictionaries
    """
    frames_data = await page.evaluate("""
        () => {
            const blocks = Array.from(document.querySelectorAll('.block'));
            return blocks.map((block, index) => {
                const rect = block.getBoundingClientRect();
                const styles = window.getComputedStyle(block);
                
                // Parse border-radius (take first value if multiple)
                const borderRadius = styles.borderRadius;
                let borderRadiusPx = 0;
                if (borderRadius && borderRadius !== 'none') {
                    const match = borderRadius.match(/(\\d+(?:\\.\\d+)?)px/);
                    if (match) {
                        borderRadiusPx = Math.round(parseFloat(match[1]));
                    }
                }
                
                // Parse z-index
                const zIndex = parseInt(styles.zIndex) || 0;
                
                // Get background color
                const bgColor = styles.backgroundColor;
                
                // Get box-shadow
                const boxShadow = styles.boxShadow || 'none';
                
                return {
                    id: index + 1,
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    w: Math.round(rect.width),
                    h: Math.round(rect.height),
                    z_index: zIndex,
                    border_radius: borderRadiusPx,
                    bg_color: bgColor,
                    box_shadow: boxShadow,
                    in_header: false,
                    in_footer: false
                };
            });
        }
    """)
    
    return frames_data


async def generate_metadata_from_page(page, page_id: int, viewport_width: int, viewport_height: int) -> dict:
    """Generate metadata by extracting frame coordinates from rendered page.
    
    Args:
        page: Playwright page object
        page_id: Page ID
        viewport_width: Viewport width used for rendering
        viewport_height: Viewport height used for rendering
        
    Returns:
        Metadata dictionary
    """
    # Get page dimensions
    page_height = await page.evaluate('document.documentElement.scrollHeight')
    
    # Extract frames metadata
    frames_metadata = await extract_frames_metadata(page)
    
    metadata = {
        'page_id': page_id,
        'page_width': viewport_width,  # Viewport width used for rendering
        'page_height': page_height,  # Actual rendered page height
        'screenshot_viewport_width': viewport_width,
        'screenshot_viewport_height': viewport_height,
        'frames': frames_metadata
    }
    
    return metadata


async def render_page(width: int, height: int, html_path: Path, output_dir: Path, meta_dir: Path = None):
    """Prepare page, scroll to end, and take screenshot.
    
    Args:
        width: Viewport width in pixels
        height: Viewport height in pixels
        html_path: Path to HTML file to render
        output_dir: Directory to save screenshot
        meta_dir: Directory with metadata JSON files (optional)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': width, 'height': height}
        )
        page = await context.new_page()
        
        # Load HTML file and wait for load event (loads all resources)
        file_url = f"file://{html_path.absolute()}"
        await page.goto(file_url, wait_until='load', timeout=30000)
        
        # Disable animations and transitions
        await page.add_style_tag(content='*{animation:none!important;transition:none!important;}')
        
        # Fix body height to allow full page rendering
        # Remove height: 100vh restriction that prevents full page rendering
        await page.add_style_tag(content='body{height:auto!important;min-height:100vh!important;}')
        
        # Wait for layout recalculation after style change
        await page.wait_for_timeout(300)
        
        # Wait for all images to load
        await page.evaluate("""
            Promise.all(
                Array.from(document.images).map(img => {
                    if (img.complete) return Promise.resolve();
                    return new Promise((resolve) => {
                        img.onload = resolve;
                        img.onerror = resolve;
                    });
                })
            )
        """)
        
        # Scroll to load all content
        viewport_height = height
        last_height = 0
        scroll_attempts = 0
        max_scroll_attempts = 50
        
        while scroll_attempts < max_scroll_attempts:
            # Get current scroll height from documentElement (more reliable)
            current_height = await page.evaluate('Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)')
            
            if current_height == last_height:
                # Height stabilized, check if we're at bottom
                scroll_y = await page.evaluate('window.scrollY || window.pageYOffset')
                max_scroll = current_height - viewport_height
                if scroll_y >= max_scroll - 10:
                    break
            
            # Scroll down
            await page.evaluate(f'window.scrollBy(0, {viewport_height})')
            await page.wait_for_timeout(200)  # Minimal wait for rendering
            
            last_height = current_height
            scroll_attempts += 1
        
        # Scroll back to top and wait for stabilization
        await page.evaluate('window.scrollTo(0, 0)')
        await page.wait_for_timeout(200)  # Minimal wait for final stabilization
        
        # Extract page_id from filename
        page_id = int(html_path.stem.replace('page_', ''))
        
        # Generate metadata from rendered page
        if meta_dir is not None:
            metadata = await generate_metadata_from_page(page, page_id, width, height)
            
            # Save metadata
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / f"page_{page_id}.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename from HTML filename
        output_filename = html_path.stem + '.png'
        output_path = output_dir / output_filename
        
        # Take full page screenshot
        await page.screenshot(path=str(output_path), full_page=True)
        
        await context.close()
        await browser.close()
    
    return output_path


def render(
    input_dir: str = "data/pages",
    output_dir: str = "data/screenshots",
    meta_dir: str = "data/meta",
    workers: int = None,
    mode: str = "random"
):
    """
    Render HTML files from input_dir to screenshots in output_dir using Playwright.

    Args:
        input_dir: Directory with HTML files.
        output_dir: Directory to save screenshots.
        meta_dir: Directory with metadata JSON files.
        workers: Number of parallel workers (default: CPU count).
        mode: Resolution mode ('phone', 'desktop', or 'random').
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    meta_dir = Path(meta_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set number of workers
    if workers is None:
        workers = cpu_count()
    
    html_files = sorted(input_dir.glob("*.html"))

    if not html_files:
        print(f"No HTML files found in {input_dir}")
        return

    async def render_all():
        # Create progress bar
        pbar = tqdm(total=len(html_files), desc="Rendering pages", unit="page")
        
        # Create semaphore to limit concurrent workers
        semaphore = asyncio.Semaphore(workers)
        
        async def render_one(html_path):
            async with semaphore:  # Limit concurrent executions
                try:
                    # Get random resolution based on mode
                    width, height = random_resolution(mode)
                    output_path = await render_page(width, height, html_path, output_dir, meta_dir)
                    pbar.set_postfix_str(f"{html_path.name} ({width}x{height})")
                except Exception as e:
                    pbar.set_postfix_str(f"Error: {html_path.name}")
                    print(f"\nError rendering {html_path}: {e}")
                finally:
                    pbar.update(1)

        # Parallel rendering with semaphore limit
        tasks = [render_one(html_path) for html_path in html_files]
        await asyncio.gather(*tasks)
        
        pbar.close()

    asyncio.run(render_all())






def main():
    args = parse_args()
    render(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        meta_dir=args.meta_dir,
        workers=args.workers,
        mode=args.mode
    )

if __name__ == "__main__":
    main()