# src/graph_constructor.py

import networkx as nx
import xml.etree.ElementTree as ET

def build_graph_from_kgml(index_by_entry: dict, xml_path: str) -> nx.DiGraph:
    """
    Construct a directed graph from the <relation> elements in a KGML file.

    Each relation in the KGML specifies an interaction between two entries
    (e.g., genes or compounds). This function builds a NetworkX DiGraph
    whose nodes correspond to the provided entry IDs and whose edges
    correspond to relations between entries that were successfully matched.

    Parameters
    ----------
    index_by_entry : dict
        A mapping from KGML entry ID (string) to its index in the matched
        gene list. Only entry IDs present here will be included as nodes.
    xml_path : str
        Filesystem path to the KGML (XML) file.

    Returns
    -------
    nx.DiGraph
        A directed graph in which nodes are entry IDs (as strings) and
        edges represent directed relations (entry1 → entry2).
    """
    # Parse the KGML XML document
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize an empty directed graph
    G = nx.DiGraph()

    # Add each matched entry ID as a node
    for entry_id in index_by_entry:
        G.add_node(entry_id)

    # Iterate over all <relation> elements and add edges accordingly
    for relation in root.findall('relation'):
        source_id = relation.get('entry1')
        target_id = relation.get('entry2')

        # Only include edges between nodes that were previously matched
        if source_id in index_by_entry and target_id in index_by_entry:
            G.add_edge(source_id, target_id)

    return G
