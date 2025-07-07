import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

# Use for debugging during dev, visualizes columns/rows/paragraphs
def debug_img(img: np.ndarray, segments: list[tuple[int, int, int, int]], color: tuple[int, int, int], out_dir: str, filename: str) -> None:
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # No y coords for columns cause info isn't known yet
    for x, y, w, h in segments:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, filename), debug_img)


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


def merge_segments(segments: List[Tuple[int, int]], max_gap: Optional[int] = None) -> List[Tuple[int, int]]:
   
    # Calculate adaptive gap if not provided
    if max_gap is None:
        max_gap = calculate_adaptive_gap(segments)

    merged = []
    current_start, current_end = segments[0] # Take the first segment start and end as current
   
   # If the NEXT segment's start doesn't exceed map_gap, merge
    for next_start, next_end in segments[1:]:
        if next_start - current_end <= max_gap:
            current_end = next_end
        else:
            # Else, start a new segment
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
   
    merged.append((current_start, current_end))
    return merged # list of paragraphs


def calculate_adaptive_gap(segments: List[Tuple[int, int]], default_gap: int = 30) -> int:

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
   
    # Sort gaps to identify a "natural" jump
    sorted_gaps = sorted(gaps)
    diffs = [b - a for a, b in zip(sorted_gaps, sorted_gaps[1:])]

    if not diffs:
        return default_gap

    # Find the largest jump between two gaps, that can differentiate line vs paragraph spacing
    max_jump_idx = np.argmax(diffs)
    adaptive_gap = sorted_gaps[max_jump_idx] #  the biggest disparity found between gaps as paragraph break

    return int(adaptive_gap)


# Save extracted paragraphs to imgs in specified output directory
def save_crop(img: np.ndarray, box: Tuple[int, int, int, int], page_id: str, para_idx: int, out_dir: str) -> None:
    x, y, w, h = box
   
    # Add small padding to improve readability
    padding = 10
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
    col_segs = find_segments(col_histogram, min_thresh=5)
    col_segs = filter_segments(col_segs, min_size=50)
    
    # debug_img(binary, [(x0, 0, x1 - x0, binary.shape[0]) for x0, x1 in col_segs], (0, 255, 0), "debug imgs/columns", f"{page_id}_col_segs.png")

    print(f"  Detected {len(col_segs)} columns")

    para_boxes = []
    total_row_segments = 0
    total_merged_segments = 0
   
   # Detects each ROW in a column
    for col_idx, (x0, x1) in enumerate(col_segs):
        col_img = binary[:, x0:x1]
       
        row_proj = np.sum(col_img > 0, axis=1)
        row_segs = find_segments(row_proj, min_thresh=5)
        row_segs = filter_segments(row_segs, min_size=5)

        # debug_img(binary, [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in row_segs], (0, 255, 255), "debug imgs/rows", f"{page_id}_col{col_idx + 1}_rows.png")
       
        # This merges the rows to form a paragraph
        merged_segs = merge_segments(row_segs)
        merged_segs = filter_segments(merged_segs, min_size=5)

        # debug_img(binary, [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in merged_segs], (255, 0, 0), "debug imgs/paragraphs", f"{page_id}_col{col_idx+1:02d}_paragraphs.png")

        total_row_segments += len(row_segs)
        total_merged_segments += len(merged_segs)
       
        print(f"    Column {col_idx + 1}: {len(row_segs)} lines → {len(merged_segs)} paragraphs")
       
        for y0, y1 in merged_segs:
            box = (x0, y0, x1 - x0, y1 - y0)
            para_boxes.append(box)

    print(f"  Total: {total_row_segments} line segments → {total_merged_segments} paragraphs\n")
   
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
   
    # Save paragraph crops
    for idx, box in enumerate(para_boxes, start=1):
        save_crop(img, box, page_id, idx, out_dir)

    return len(para_boxes)


def main():
    input_dir = "example imgs"
    output_dir = "outputs"
   
    # Find PNG files
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
   
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".png")]
    print(f"Found {len(files)} PNG files: {files}\n")
   
    if not files:
        print("No PNG files found in input directory")
        return

    # Process all pages
    total_paragraphs = 0
    for fname in files:
        img_path = os.path.join(input_dir, fname)
        para_count = extract_paragraphs(img_path, output_dir)
        total_paragraphs += para_count
   
    print(f"Total paragraphs extracted: {total_paragraphs}")
    print(f"Average paragraphs per page: {total_paragraphs / len(files):.1f}")


if __name__ == "__main__":
    main()