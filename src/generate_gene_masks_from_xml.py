#!/usr/bin/env python3
"""
generate_gene_masks_from_xml.py

From data/xml/{train,test}/… XMLs, generate binary masks
under data/masks/genes/{train,test}/… matching the same IDs.
"""
import logging
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BASE = Path(__file__).resolve().parent.parent / "data"
SETS = ["train", "test"]

for subset in SETS:
    xml_dir  = BASE / "xml"   / subset
    img_dir  = BASE / "images"/ subset
    mask_dir = BASE / "masks" / "genes"   / subset
    mask_dir.mkdir(parents=True, exist_ok=True)

    for xml_file in xml_dir.glob("*.xml"):
        pid = xml_file.stem
        img_file = img_dir / f"{pid}.png"
        if not img_file.exists():
            logging.warning(f"[{subset}] Missing image for {pid}")
            continue

        # get dimensions
        img = Image.open(img_file)
        w, h = img.size
        mask = np.zeros((h, w), dtype=np.uint8)

        # parse KGML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        entries = root.findall(".//entry") or root.findall(".//{*}entry")

        for e in entries:
            if e.get("type") != "gene":
                continue
            g = e.find("graphics") or e.find("{*}graphics")
            if g is None:
                continue
            try:
                x, y = float(g.get("x")), float(g.get("y"))
                ww, hh = float(g.get("width")), float(g.get("height"))
            except:
                continue

            # convert center→corners
            x1 = max(0, int(x - ww/2))
            y1 = max(0, int(y - hh/2))
            x2 = min(w,  int(x + ww/2))
            y2 = min(h,  int(y + hh/2))
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)

        out_path = mask_dir / f"{pid}_mask.png"
        cv2.imwrite(str(out_path), mask)
        logging.info(f"[{subset}] Saved mask → {out_path}")
