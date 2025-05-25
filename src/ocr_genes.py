import cv2
import numpy as np
import pytesseract
import easyocr
from typing import List, Tuple

OCR_SCALE = 2
MIN_AREA = 200
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_EASYOCR_READER = easyocr.Reader(['en'], gpu=False)

def extract_gene_names(
    image_path: str,
    box_coordinates: List[Tuple[int, int, int, int]],
    method: str = "easyocr"
) -> List[str]:
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

        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"OCR Box {idx}: skipped (empty region after cropping)")
            results.append("")
            continue

        h2, w2 = roi.shape[:2]
        roi = cv2.resize(
            roi,
            (w2 * OCR_SCALE, h2 * OCR_SCALE),
            interpolation=cv2.INTER_LINEAR
        )

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        try:
            if method.lower() == "tesseract":
                text = pytesseract.image_to_string(
                    roi_rgb,
                    config=_TESSERACT_CONFIG
                ).strip()
                print(f"OCR Box {idx} (Tesseract): [{text}]")
            elif method.lower() == "easyocr":
                texts = _EASYOCR_READER.readtext(roi_rgb, detail=0)
                text = texts[0] if texts else ""
                print(f"OCR Box {idx} (EasyOCR): [{text}]")
            else:
                raise ValueError("Unsupported OCR method: choose 'tesseract' or 'easyocr'")
        except Exception as ex:
            print(f"OCR Box {idx}: OCR error: {ex}")
            text = ""

        results.append(text or "")

    return results
