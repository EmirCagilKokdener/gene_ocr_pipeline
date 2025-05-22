import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(graph: nx.DiGraph, save_path: str = None, figsize: tuple = (8, 6)):
    """
    Render and optionally save a visualization of a directed graph using a spring layout.

    This function computes node positions via a force-directed (spring) layout,
    draws nodes and directed edges with arrowheads, and annotates each node with its label.

    Args:
        graph (nx.DiGraph): A directed graph containing nodes and edges to visualize.
        save_path (str, optional): File path to save the rendered figure. If None, the plot is displayed interactively.
        figsize (tuple, optional): Width and height of the figure in inches. Default is (8, 6).
    """
    # Initialize the figure with specified dimensions
    plt.figure(figsize=figsize)

    # Compute spatial positions for nodes using a reproducible spring layout
    pos = nx.spring_layout(graph, seed=42)

    # Draw nodes: light blue circles with black outlines
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=300,
        node_color='lightblue',
        edgecolors='black'
    )

    # Draw directed edges with arrowheads and slight curvature
    nx.draw_networkx_edges(
        graph,
        pos,
        arrowstyle='-|>',
        arrowsize=12,
        connectionstyle='arc3,rad=0.1'
    )

    # Annotate nodes with their identifiers
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=8
    )

    # Remove axis for a clean presentation
    plt.axis('off')

    # Save or display the figure based on save_path
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
