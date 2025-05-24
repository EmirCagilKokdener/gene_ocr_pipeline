import json
import argparse
from pathlib import Path

# Determine the project root directory
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
IMG_DIR     = DATA_DIR / "images" / "test"
XML_DIR     = DATA_DIR / "xml" / "test"
RESULTS_DIR = BASE_DIR / "data" / "results" / "pipeline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline components
from segment_genes      import segment_genes
from ocr_genes          import extract_gene_names
from match_with_kgml    import match_ocr_to_kgml
from graph_constructor  import build_graph_from_kgml
from visualize_graph    import visualize_graph


def process_pathway(pathway_id: str):
    """
    Execute the complete KEGG pathway processing pipeline for a single pathway.

    Parameters
    ----------
    pathway_id : str
        KEGG pathway identifier (e.g., 'hsa04110').

    Pipeline steps
    --------------
    1) Segment gene regions within the pathway image.
    2) Extract gene names from each region using EasyOCR.
    3) Match OCR results to KGML entries.
    4) Persist intermediate results to JSON.
    5) Construct a directed graph from KGML relations.
    6) Render and save the graph visualization.
    """
    print(f"\n=== Processing pathway: {pathway_id} (OCR engine: EasyOCR only) ===")

    img_path = IMG_DIR / f"{pathway_id}.png"
    xml_path = XML_DIR / f"{pathway_id}.xml"

    if not img_path.exists():
        print(f"[!] Pathway image not found: {img_path}")
        return
    if not xml_path.exists():
        print(f"[!] Pathway XML not found: {xml_path}")
        return

    # Step 1: Segment genes
    gene_boxes = segment_genes(str(img_path))

    # Step 2: OCR via EasyOCR
    gene_names = extract_gene_names(str(img_path), gene_boxes)

    # Step 3: Match to KGML
    kgml_result    = match_ocr_to_kgml(gene_names, str(xml_path))
    matches        = kgml_result.get("matches", [])
    index_by_entry = kgml_result.get("index_by_entry", {})

    # Step 4: Save JSON
    output_data = {
        "gene_boxes":     gene_boxes,
        "gene_names":     matches,
        "index_by_entry": index_by_entry
    }
    out_file = RESULTS_DIR / f"{pathway_id}_pipeline.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Pipeline outputs saved to: {out_file}")

    # Step 5: Build graph
    G = build_graph_from_kgml(index_by_entry, str(xml_path))

    # Step 6: Visualize graph
    graph_image = RESULTS_DIR / f"{pathway_id}_graph.png"
    visualize_graph(G, save_path=str(graph_image))
    print(f"Graph visualization saved to: {graph_image}")


def main():
    """
    Parse command-line arguments and invoke the processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Execute the full KEGG pathway processing pipeline (EasyOCR only)"
    )
    parser.add_argument(
        "--pathway",
        required=True,
        help="Target KEGG pathway ID (e.g., 'hsa04110')"
    )
    args = parser.parse_args()

    process_pathway(args.pathway)


if __name__ == "__main__":
    main()
