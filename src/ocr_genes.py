import cv2
import numpy as np
import easyocr
from typing import List, Tuple

# OCR configuration
OCR_SCALE = 2      # Upscaling factor to enhance character legibility
MIN_AREA  = 200    # Minimum allowable area for a bounding box to undergo OCR

# Initialize the EasyOCR reader
_EASYOCR_READER = easyocr.Reader(['en'], gpu=False)

def extract_gene_names(
    image_path: str,
    box_coordinates: List[Tuple[int, int, int, int]]
) -> List[str]:
    """
    Recognize gene labels within specified bounding boxes using EasyOCR.

    For each box:
      1. Skip if the box area is below MIN_AREA.
      2. Apply a small padding to capture full characters.
      3. Upscale the crop by OCR_SCALE for improved text resolution.
      4. Convert from BGR to RGB as required by EasyOCR.
      5. Return only the first recognized text, or "" if none.

    Args:
        image_path (str): Path to the input image.
        box_coordinates (List[Tuple[int,int,int,int]]):
            List of (x, y, width, height) bounding boxes.

    Returns:
        List[str]: The top OCR result for each box, or "" if nothing recognized.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image at: {image_path}")

    results: List[str] = []
    for idx, (x, y, w, h) in enumerate(box_coordinates):
        area = w * h
        if area < MIN_AREA:
            print(f"OCR Box {idx}: skipped (area {area} < {MIN_AREA})")
            results.append("")
            continue

        # Apply padding and clamp to image bounds
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"OCR Box {idx}: skipped (empty region after cropping)")
            results.append("")
            continue

        # Upscale for better OCR accuracy
        h2, w2 = roi.shape[:2]
        roi = cv2.resize(
            roi,
            (w2 * OCR_SCALE, h2 * OCR_SCALE),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to RGB and run EasyOCR
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        try:
            texts = _EASYOCR_READER.readtext(roi_rgb, detail=0)
        except Exception as ex:
            print(f"OCR Box {idx}: EasyOCR error: {ex}")
            texts = []

        print(f"OCR Box {idx}: {texts}")
        results.append(texts[0] if texts else "")

    return results
