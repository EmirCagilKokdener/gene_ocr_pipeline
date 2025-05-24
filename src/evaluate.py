# src/evaluate.py
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def evaluate(pathway_id: str):
    BASE_DIR = Path(__file__).resolve().parent.parent
    xml_path = BASE_DIR / "data/xml/test" / f"{pathway_id}.xml"
    pipeline_path = BASE_DIR / "data/results/pipeline" / f"{pathway_id}_pipeline.json"

    # 1) ground-truth gene entry IDs from KGML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = { e.get("id")
           for e in root.findall("entry")
           if e.get("type") == "gene" }

    # 2) predicted entry IDs
    with open(pipeline_path, encoding="utf-8") as f:
        data = json.load(f)
    preds = set(data.get("index_by_entry", {}).keys())

    # 3) compute counts
    tp = len(preds & gt)
    fp = len(preds - gt)
    fn = len(gt - preds)
    tn = 0  # we only care about the gene class here

    # 4) metrics
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    accuracy  = tp / (tp + fn + fp) if (tp + fn + fp) else 0  # positive accuracy

    # 5) report
    print(f"Pathway: {pathway_id}")
    print(f"  GT genes:      {len(gt)}")
    print(f"  Predicted:     {len(preds)}")
    print(f"  True positives:{tp}")
    print(f"  False negatives:{fn}")
    print(f"  False positives:{fp}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pathway", help="e.g. hsa04110")
    args = p.parse_args()
    evaluate(args.pathway)
