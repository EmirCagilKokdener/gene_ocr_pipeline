#!/usr/bin/env python3
"""
download_kegg.py

Downloads KGML and PNG files for two sets of KEGG pathways:
  - TRAIN_IDS → data/xml/train/ & data/images/train/
  - TEST_IDS  → data/xml/test/  & data/images/test/
"""

import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Split your 10+2 IDs
# in download_kegg.py
TRAIN_IDS = [
    "hsa04110", "hsa04115", "hsa04210", "hsa03430",
    "hsa04310", "hsa04330", "hsa04010", "hsa04630",
    "hsa04152", "hsa04910"
]
TEST_IDS = [
    "hsa04350", "hsa04151"
]


def download_set(ids, xml_dir: Path, img_dir: Path):
    xml_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    for pid in ids:
        # KGML
        xml_url = f"http://rest.kegg.jp/get/{pid}/kgml"
        xml_path = xml_dir / f"{pid}.xml"
        try:
            r = requests.get(xml_url, timeout=10)
            r.raise_for_status()
            xml_path.write_text(r.text, encoding="utf-8")
            logging.info(f"Downloaded KGML for {pid} → {xml_path}")
        except Exception as e:
            logging.error(f"KGML download failed for {pid}: {e}")

        # PNG
        img_url = f"http://rest.kegg.jp/get/{pid}/image"
        img_path = img_dir / f"{pid}.png"
        try:
            r = requests.get(img_url, timeout=10)
            r.raise_for_status()
            img_path.write_bytes(r.content)
            logging.info(f"Downloaded image for {pid} → {img_path}")
        except Exception as e:
            logging.error(f"Image download failed for {pid}: {e}")

def main():
    base = Path(__file__).resolve().parent.parent / "data"
    # TRAIN
    download_set(
        TRAIN_IDS,
        xml_dir = base / "xml"   / "train",
        img_dir = base / "images"/ "train"
    )
    # TEST
    download_set(
        TEST_IDS,
        xml_dir = base / "xml"   / "test",
        img_dir = base / "images"/ "test"
    )

if __name__ == "__main__":
    main()
