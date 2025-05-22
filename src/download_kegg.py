#!/usr/bin/env python3
"""
download_kegg.py

Downloads KGML and PNG files for a given list of KEGG pathway IDs,
saving them under the project’s top-level data/xml and data/images folders.
"""

import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    # Determine project directories
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    xml_dir = data_dir / "xml"
    img_dir = data_dir / "images"

    # Create directories if they do not exist
    for d in (xml_dir, img_dir):
        d.mkdir(parents=True, exist_ok=True)

    # List of KEGG pathway IDs to download
    pathway_ids = [
        "hsa04110", "hsa04115", "hsa04210", "hsa03430", "hsa04310",
        "hsa04330", "hsa04010", "hsa04630", "hsa04350", "hsa04151"
    ]

    for pid in pathway_ids:
        # Download the KGML (XML) file
        xml_url = f"http://rest.kegg.jp/get/{pid}/kgml"
        xml_path = xml_dir / f"{pid}.xml"
        try:
            response = requests.get(xml_url, timeout=10)
            response.raise_for_status()
            xml_path.write_text(response.text, encoding="utf-8")
            logging.info(f"Downloaded KGML for {pid} → {xml_path}")
        except Exception as e:
            logging.error(f"Failed to download KGML for {pid}: {e}")

        # Download the pathway image
        img_url = f"http://rest.kegg.jp/get/{pid}/image"
        img_path = img_dir / f"{pid}.png"
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            img_path.write_bytes(response.content)
            logging.info(f"Downloaded image for {pid} → {img_path}")
        except Exception as e:
            logging.error(f"Failed to download image for {pid}: {e}")

if __name__ == "__main__":
    main()
