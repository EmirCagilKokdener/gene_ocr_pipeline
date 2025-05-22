#!/usr/bin/env python3
"""
generate_gene_masks_from_xml.py

Parses KGML (XML) files to generate binary masks for gene entries
based on their graphic coordinates. Outputs one mask per pathway
under data/masks/genes/.
"""

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR  = BASE_DIR / "data" / "images"
XML_DIR  = BASE_DIR / "data" / "xml"
MASK_DIR = BASE_DIR / "data" / "masks" / "genes"
MASK_DIR.mkdir(parents=True, exist_ok=True)

def generate_gene_masks_from_kgml():
    """
    Iterate over all KGML XML files and create a binary mask highlighting
    gene rectangles. Masks are saved to data/masks/genes/{pathway_id}_mask.png.
    """
    for xml_file in XML_DIR.glob("*.xml"):
        pathway_id = xml_file.stem
        img_file = IMG_DIR / f"{pathway_id}.png"

        if not img_file.exists():
            logging.warning(f"Image not found for {pathway_id}: {img_file}")
            continue

        # Load the original image to get its dimensions
        image = Image.open(img_file)
        width, height = image.size

        # Initialize an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Parse the KGML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find all <entry> elements, accounting for possible namespaces
        entries = root.findall(".//entry") or root.findall(".//{*}entry")

        for entry in entries:
            if entry.get("type") != "gene":
                continue

            # Find the graphics element (namespace-agnostic)
            graphics = entry.find("graphics") or entry.find("{*}graphics")
            if graphics is None:
                continue

            # Read and convert coordinates
            try:
                x = float(graphics.get("x"))
                y = float(graphics.get("y"))
                w = float(graphics.get("width"))
                h = float(graphics.get("height"))
            except (TypeError, ValueError):
                continue

            # Convert center+size to top-left/bottom-right
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(width,  int(x + w / 2))
            y2 = min(height, int(y + h / 2))

            # Fill the rectangle on the mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)

        # Save the mask
        out_path = MASK_DIR / f"{pathway_id}_mask.png"
        cv2.imwrite(str(out_path), mask)
        logging.info(f"Saved gene mask for {pathway_id} → {out_path}")

if __name__ == "__main__":
    generate_gene_masks_from_kgml()
