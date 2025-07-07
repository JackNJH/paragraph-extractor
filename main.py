import os
import cv2
import config
import numpy as np
from typing import List, Tuple, Optional

# Define segment areas based on 1D histogram given
def find_segments(hist: np.ndarray, min_thresh: int = 1) -> List[Tuple[int, int]]:
    segments = []
    in_seg = False
    start = 0
   
    for idx, val in enumerate(hist):
        if val > min_thresh and not in_seg:
            start = idx
            in_seg = True
        elif val <= min_thresh and in_seg:
            segments.append((start, idx))
            in_seg = False
   
    if in_seg:
        segments.append((start, len(hist)))
   
    return segments

# This is REQUIRED to remove certain weird pickups where dangling descenders (e.g. p, y, g) gets registered as a row
def filter_segments(segments: List[Tuple[int, int]], min_size: int = 15) -> List[Tuple[int, int]]:
    return [(start, end) for start, end in segments if end - start >= min_size]


def calculate_adaptive_gap(segments: List[Tuple[int, int]], default_gap: int = 20) -> int:
    """
    Calculate adaptive gap threshold based on typical line spacing in the document.
    """
    if len(segments) < 2:
        return default_gap
   
    # Calculate gaps between consecutive segments
    gaps = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1][0] - segments[i][1]
        if gap > 0:
            gaps.append(gap)
   
    if not gaps:
        return default_gap
   
    # Use median gap as base, with reasonable bounds
    median_gap = np.median(gaps)
    adaptive_gap = max(min(median_gap * 1.5, 50), 10)  # Between 10-50 pixels
   
    return int(adaptive_gap)


def merge_segments(segments: List[Tuple[int, int]], max_gap: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Merge 1D segments that are separated by no more than max_gap.
    If max_gap is None, calculate it adaptively.
    """
    if not segments:
        return []
   
    # Sort segments by start position
    segments = sorted(segments)
   
    # Calculate adaptive gap if not provided
    if max_gap is None:
        max_gap = calculate_adaptive_gap(segments)
   
    merged = []
    current_start, current_end = segments[0]
   
    for start, end in segments[1:]:
        if start - current_end <= max_gap:
            # Extend current segment
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
   
    merged.append((current_start, current_end))
    return merged



def save_crop(img: np.ndarray, box: Tuple[int, int, int, int], page_id: str, para_idx: int, out_dir: str) -> None:
    """
    Crop a paragraph region and save as an image file.
    """
    x, y, w, h = box
   
    # Add small padding to improve readability
    padding = 5
    x_pad = max(0, x - padding)
    y_pad = max(0, y - padding)
    w_pad = min(img.shape[1] - x_pad, w + 2 * padding)
    h_pad = min(img.shape[0] - y_pad, h + 2 * padding)
   
    crop = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
    filename = f"{page_id}_para{para_idx:02d}.png"
    filepath = os.path.join(out_dir, filename)
   
    success = cv2.imwrite(filepath, crop)
    if not success:
        print(f"    Warning: Failed to save {filename}")


# Process an img and returns the number of paragraphs extracted
def extract_paragraphs(img_path: str, out_dir: str) -> int:

    page_id = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing page: {page_id}")

    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"  Error: Could not load image {img_path}")
        return 0
   
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect blocks of paragraphs first before counting 'lines' in each
    col_histogram = np.sum(binary > 0, axis=0)
    col_segs = find_segments(col_histogram, min_thresh=10)
    col_segs = filter_segments(col_segs, min_size=50)
   
    print(f"  Detected {len(col_segs)} columns")

    para_boxes = []
    total_row_segments = 0
    total_merged_segments = 0
   
    for col_idx, (x0, x1) in enumerate(col_segs):
        col_img = binary[:, x0:x1]
       
        # Line detection
        row_proj = np.sum(binary > 0, axis=1)
        row_segs = find_segments(row_proj, min_thresh=8)
        row_segs = filter_segments(row_segs, min_size=5)
       
        # Adaptive merging
        merged_segs = merge_segments(row_segs)
        merged_segs = filter_segments(merged_segs, min_size=20)
       
        total_row_segments += len(row_segs)
        total_merged_segments += len(merged_segs)
       
        print(f"    Column {col_idx + 1}: {len(row_segs)} lines → {len(merged_segs)} paragraphs")
       
        for y0, y1 in row_segs:
            box = (x0, y0, x1 - x0, y1 - y0)
            para_boxes.append(box)

    print(f"  Total: {total_row_segments} line segments → {total_merged_segments} paragraphs")
    print(f"  Final paragraph boxes: {len(para_boxes)} (after table filtering)")

    # Sort paragraphs in reading order (left-to-right, top-to-bottom)
    para_boxes.sort(key=lambda b: (b[0] // 100, b[1]))  # Group by approximate column, then by y-position
   
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
   
    # Save paragraph crops
    for idx, box in enumerate(para_boxes, start=1):
        save_crop(img, box, page_id, idx, out_dir)

    print(f"  Saved {len(para_boxes)} paragraphs for {page_id}\n")
    return len(para_boxes)


def main():
    print(f"Enhanced Paragraph Extraction")
    print(f"Input: {config.input_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Min paragraph size: {config.min_para_size}px")
   
    # Find PNG files
    if not os.path.exists(config.input_dir):
        print(f"Error: Input directory '{config.input_dir}' does not exist")
        return
   
    files = [f for f in sorted(os.listdir(config.input_dir)) if f.lower().endswith(".png")]
    print(f"Found {len(files)} PNG files: {files}\n")
   
    if not files:
        print("No PNG files found in input directory")
        return

    # Process all pages
    total_paragraphs = 0
    for fname in files:
        img_path = os.path.join(config.input_dir, fname)
        para_count = extract_paragraphs(img_path, config.output_dir)
        total_paragraphs += para_count
   
    print(f"Total paragraphs extracted: {total_paragraphs}")
    print(f"Average paragraphs per page: {total_paragraphs / len(files):.1f}")


if __name__ == "__main__":
    main()