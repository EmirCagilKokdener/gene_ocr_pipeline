# src/match_with_kgml.py

from difflib import get_close_matches
from xml.etree import ElementTree as ET

def match_ocr_to_kgml(ocr_list, xml_path):
    """
    Aligns a list of OCR-derived labels to KEGG entries by
    matching against the 'graphics@name' synonyms in the KGML file.

    For each OCR token, the function searches for the closest
    synonym within the set of all entry aliases (using fuzzy matching),
    then records both the matched synonym sequence and an index
    mapping from the KEGG entry ID to the position in the results list.

    Args:
        ocr_list (List[str]): Textual labels extracted via OCR.
        xml_path (str): Path to the KEGG KGML file.

    Returns:
        dict: {
            "matches": List[str] of the resolved synonyms (e.g., ["P53", "CDK1", ...]),
            "index_by_entry": Dict[str, int] mapping entry_id → match index
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Construct a lookup from each uppercase synonym to its entry ID
    syn2eid = {}
    for entry in root.findall('entry'):
        eid = entry.get('id')
        graphics = entry.find('graphics')
        if graphics is None or 'name' not in graphics.attrib:
            continue
        for synonym in graphics.attrib['name'].split(','):
            syn = synonym.strip().upper()
            if syn:
                syn2eid[syn] = eid

    candidates = list(syn2eid.keys())
    matches = []
    index_by_entry = {}

    for idx, label in enumerate(ocr_list):
        label_up = label.upper()
        closest = get_close_matches(label_up, candidates, n=1, cutoff=0.6)
        if not closest:
            continue
        resolved_syn = closest[0]
        matched_eid = syn2eid[resolved_syn]
        matches.append(resolved_syn)
        index_by_entry[matched_eid] = len(matches) - 1

    # Debug output
    print("KGML matches:", matches)
    print("Index by entry:", index_by_entry)

    return {
        "matches": matches,
        "index_by_entry": index_by_entry
    }
