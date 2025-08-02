import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

# Used for debugging during development, visualizes columns/rows/paragraphs
def debug_img(img: np.ndarray, segments: list[tuple[int, int, int, int]], color: tuple[int, int, int], out_dir: str, filename: str) -> None:
    
    # Convert image to BGR for colored lines
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw each provided rectangle on to the image
    for x, y, w, h in segments:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)

    # Prepare debug image output directory (e.g., "debug imgs/columns")
    debug_out_dir = os.path.join("debug imgs", out_dir)
    os.makedirs(debug_out_dir, exist_ok=True)

    # Save the images to a separate debug folder
    cv2.imwrite(os.path.join(debug_out_dir, filename), debug_img)


# Define segment areas based on 1D histogram (row or column projection)
def find_segments(hist: np.ndarray, min_thresh: int = 1) -> List[Tuple[int, int]]:
    segments = []       # stores the start and end pixel position of each content segment
    in_seg = False      # flag to check if we are inside a segment currently
    start = 0           # variable to remember start index of a segment
   
    # Scanning through list to find regions of non-zero values (meaning these are content segments)
    for idx, val in enumerate(hist):

        # If value is higher than threshold and in_seg status = false -> start a segment
        if val > min_thresh and not in_seg:
            start = idx
            in_seg = True
        # If value drops below threshold -> segment ends
        elif val <= min_thresh and in_seg:
            segments.append((start, idx))
            in_seg = False
   
    # If the image ends while still in_seg = true, add to the last index
    if in_seg:
        segments.append((start, len(hist)))
   
    return segments


# Remove segments that are too small to be meaningful
# (This is REQUIRED to remove certain weird pickups where iffy noise gets registered as a row)
def filter_segments(segments: List[Tuple[int, int]], min_size: int = 15) -> List[Tuple[int, int]]:
    return [(start, end) for start, end in segments if end - start >= min_size] # only keep segments where the height/width is at least 'min_size'


# Merge segments that are close together based on max allowed gap 
def merge_segments(segments: List[Tuple[int, int]], max_gap: Optional[int] = None) -> List[Tuple[int, int]]:
   
    # Calculate adaptive gap if not provided
    if max_gap is None:
        max_gap = calculate_adaptive_gap(segments)

    merged = [] # final list of merged segments
    current_start, current_end = segments[0] # take the first segment's start and end as current points
   
   # If the NEXT segment's start doesn't exceed map_gap, merge them together
    for next_start, next_end in segments[1:]:
        if next_start - current_end <= max_gap:
            # Extend current segment
            current_end = next_end
        else:
            # Else, store current points and move on to the next
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
   
    # Append final segment
    merged.append((current_start, current_end))
    return merged


# Dynamically determine what "gap" qualifies as paragraph break
def calculate_adaptive_gap(segments: List[Tuple[int, int]], default_gap: int = 30) -> int:

    if len(segments) < 2:
        return default_gap # Not enough segment samples to guess adaptive value
   
    gaps = [] # list of all distance between segments

    # Find all gaps between segments by taking end of segment(1) - start of segment(2)
    for i in range(len(segments) - 1): 
        gap = segments[i + 1][0] - segments[i][1]
        if gap > 0:
            gaps.append(gap)
   
    # If all segments were overlapping or adjacent, fallback to default
    if not gaps:
        return default_gap
   
    # Sort gaps to identify a "natural" jump
    sorted_gaps = sorted(gaps)

    # Compute the DIFFERENCES between each consecutive gap
    # Core idea: if there's a sudden big jump in gap, we will catch that later and use it as the gap between paragraphs
    diffs = [b - a for a, b in zip(sorted_gaps, sorted_gaps[1:])]

    if not diffs:
        return default_gap

    # Find the largest jump between two gaps, that can differentiate line vs paragraph spacing
    max_jump_idx = np.argmax(diffs)
    adaptive_gap = sorted_gaps[max_jump_idx] # the biggest disparity found between gaps as paragraph break

    return int(adaptive_gap)


# Save cropped image based on a bounding box (paragraph area)
def save_to_img(img: np.ndarray, box: Tuple[int, int, int, int], page_id: str, para_idx: int, out_dir: str) -> None:
    x, y, w, h = box
   
    # Add small padding to improve readability
    padding = 10
    x_pad = max(0, x - padding)
    y_pad = max(0, y - padding)
    w_pad = min(img.shape[1] - x_pad, w + 2 * padding)
    h_pad = min(img.shape[0] - y_pad, h + 2 * padding)
   
    # Crop padded paragraph region
    crop = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]

    # Construct output filename and path
    filename = f"{page_id}_para{para_idx:02d}.png"
    filepath = os.path.join(out_dir, filename)
   
    # Write image to file; throw error if failure
    success = cv2.imwrite(filepath, crop)
    if not success:
        print(f"    Warning: Failed to save {filename}")


# Main logic to extract paragraphs from a single page image
def extract_paragraphs(img_path: str, out_dir: str) -> int:

    # Get file name without its extension as page ID
    page_id = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing page: {page_id}")

    # Load image in grayscale to scan pixels
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"  Error: Could not load image {img_path}")
        return 0
   
    # Convert image to binary: text becomes white (255), background becomes black (0)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # --------- Step 1: Detect columns ---------------
    col_histogram = np.sum(binary > 0, axis=0)              # count non-black pixels column-wise
    col_segs = find_segments(col_histogram, min_thresh=5)   # find continuous column segments
    col_segs = filter_segments(col_segs, min_size=50)       # remove noise

    # Line used for debugging during column detection
    # debug_img(binary, [(x0, 0, x1 - x0, binary.shape[0]) for x0, x1 in col_segs], (0, 255, 0), "columns", f"{page_id}_col_segs.png")

    print(f"  Detected {len(col_segs)} columns")

    para_boxes = []             # list of paragraphs
    total_row_segments = 0      # count of line detections (before merging)
    total_merged_segments = 0   # count of merged segments (paragraphs)


    # --------- Step 2: For each column, detect rows ---------------
    for col_idx, (x0, x1) in enumerate(col_segs):

        # Extract current column from image
        col_img = binary[:, x0:x1]
       
        row_proj = np.sum(col_img > 0, axis=1)              # count non-black pixels row-wise
        row_segs = find_segments(row_proj, min_thresh=5)    # find continuous lines
        row_segs = filter_segments(row_segs, min_size=5)    # remove noise

        # Line used for debugging during row detection
        # debug_img(binary, [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in row_segs], (0, 255, 255), "rows", f"{page_id}_col{col_idx + 1}_rows.png")
       
        
        # --------- Step 3: Merge rows into paragraphs ---------------
        merged_segs = merge_segments(row_segs)                  # merge each line into paragraphs
        merged_segs = filter_segments(merged_segs, min_size=5)  # remove noise

        # Line used for debugging when merging lines
        # debug_img(binary, [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in merged_segs], (255, 0, 0), "paragraphs", f"{page_id}_col{col_idx+1:02d}_paragraphs.png")

        total_row_segments += len(row_segs)
        accepted_count = 0  # counter for successful paragraphs processed
        para_idx = 1        # stores paragraph number in this column
       
        print(f"    Column {col_idx + 1}: {len(row_segs)} lines → {len(merged_segs)} paragraphs")
       

        # --------- Step 4: Process and save paragraphs ---------------
        for y0, y1 in merged_segs:
            box = (x0, y0, x1 - x0, y1 - y0) # box = (x, y, width, height)

            # OPTIONAL filter that filters tables/images. Commented out for sake of assignment. 
            # crop = binary[y0:y1, x0:x1]  # crop paragraph area from image
            # h, w = crop.shape
            # pixel_density = np.sum(crop > 0) / (w * h)

            # # This works cause pictures would have higher pixel density and tables would have lower (due to spacing in cells)
            # if pixel_density > 0.3 or pixel_density < 0.11:
            #     print(f"    Skipping paragraph {para_idx:02d} in column {col_idx + 1} (likely to be image/table)")
            # else:

            para_boxes.append(box)
            accepted_count += 1
            para_idx += 1 # move on to next paragraph
        
        total_merged_segments += accepted_count # update total

    print(f"  Total: {total_row_segments} line segments → {total_merged_segments} paragraphs\n")
   
    # --------- Step 5: Output all paragraphs to directory ---------------
    os.makedirs(out_dir, exist_ok=True) # make sure output folder exists

    # Save to image for each paragraph extracted in current page
    for idx, box in enumerate(para_boxes, start=1):
        save_to_img(img, box, page_id, idx, out_dir)

    return len(para_boxes)


def main():
    input_dir = "Converted Paper (8)"   # folder where input PNG files are located
    output_dir = "outputs"              # folder to save output paragraph images
   
    # Find PNG files
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
   
    # List PNG files sorted alphabetically
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".png")]
    print(f"Found {len(files)} PNG files: {files}\n")
   
    if not files:
        print("No PNG files found in input directory")
        return

    total_paragraphs = 0 # global counter for all paragraphs found

    # Start the paragraph extraction process on each image file
    for fname in files:
        img_path = os.path.join(input_dir, fname)
        para_count = extract_paragraphs(img_path, output_dir)
        total_paragraphs += para_count
   
    print(f"Total paragraphs extracted: {total_paragraphs}")
    print(f"Average paragraphs per page: {total_paragraphs / len(files):.1f}")


if __name__ == "__main__":
    main()