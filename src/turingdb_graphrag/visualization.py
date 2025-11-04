from pyvis.network import Network
from turingdb_graphrag.subgraph import get_subgraph_around_query


def visualize_graph_with_pyvis(
    subgraph,
    output_file="graph.html",
    color_map=None,
    node_label_func=None,
    node_size_func=None,
):
    """
    Visualize a NetworkX graph using PyVis.

    Args:
        subgraph: NetworkX graph to visualize
        output_file: HTML file to save
        color_map: Dict of {node_type: color} (optional)
        node_label_func: Function(node_id, node_data) -> label (optional)
        node_size_func: Function(node_id, node_data) -> size (optional)

    Returns:
        IFrame for Jupyter display (if available)
    """

    # Default color scheme
    if color_map is None:
        color_map = {
            "control": "#ff6b6b",
            "domain": "#4ecdc4",
            "topic": "#95e1d3",
            "standard": "#ffe66d",
        }

    # Create PyVis network
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )

    # Configure physics
    net.set_options(
        """
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        }
      }
    }
    """
    )

    # Add nodes
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        node_type = node_data.get("type", "unknown")

        # Color
        color = color_map.get(node_type, "#cccccc")

        # Size (custom or based on relevance/combined_score)
        if node_size_func:
            size = node_size_func(node, node_data)
        else:
            relevance = node_data.get("relevance", node_data.get("combined_score", 0))
            if relevance > 0:
                size = 30 + (relevance * 50)
            else:
                size = 20

        border_width = 3 if relevance > 0 else 1

        # Label (custom or default)
        if node_label_func:
            label = node_label_func(node, node_data)
        else:
            if node_type == "control":
                label = f"Control\n{node.split('_')[1] if '_' in str(node) else node}"
            elif node_type in ["topic", "domain"]:
                label = node_data.get("name", str(node))[:20]
            elif node_type == "standard":
                label = (
                    f"{node_data.get('standard', '')}\n{node_data.get('reference', '')}"
                )
            else:
                label = str(node)[:20]

        # Hover text
        title_parts = [f"<b>{node_type.title()}</b>"]

        # Add key attributes to hover
        for key in ["statement", "name"]:
            if key in node_data:
                text = str(node_data[key])[:200]
                title_parts.append(f"{text}...")
                break

        # Add similarity scores if present
        if "relevance" in node_data:
            title_parts.append(f"<b>Relevance: {node_data['relevance']:.3f}</b>")
        if "combined_score" in node_data:
            title_parts.append(f"<b>Combined: {node_data['combined_score']:.3f}</b>")
            title_parts.append(
                f"Structural: {node_data.get('structural_similarity', 0):.3f}"
            )
            title_parts.append(
                f"Semantic: {node_data.get('semantic_similarity', 0):.3f}"
            )
        if "is_seed" in node_data and node_data["is_seed"]:
            title_parts.append("<b>🎯 SEED NODE</b>")

        title = "<br>".join(title_parts)

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
            borderWidthSelected=5,
            font={"size": 12 if relevance > 0 else 10},
        )

    # Add edges
    for source, target in subgraph.edges():
        edge_data = subgraph.get_edge_data(source, target)
        relationship = edge_data.get("rel", "") if edge_data else ""

        net.add_edge(
            source, target, title=relationship, arrows="to", color="#888888", width=1
        )

    # Legend
    legend_html = """
    <div style='position: absolute; top: 10px; right: 10px;
                background: white; padding: 10px; border: 1px solid #ccc;
                border-radius: 5px; font-family: Arial;'>
        <b>Node Types:</b><br>
        <span style='color: #ff6b6b;'>● Control</span><br>
        <span style='color: #95e1d3;'>● Topic</span><br>
        <span style='color: #4ecdc4;'>● Domain</span><br>
        <span style='color: #ffe66d;'>● Standard</span><br>
        <br><b>Size:</b> Relevance/Score
    </div>
    """

    # Save and display
    net.save_graph(output_file)

    # Inject legend
    with open(output_file, "r") as f:
        html = f.read()
    html = html.replace("</body>", f"{legend_html}</body>")
    with open(output_file, "w") as f:
        f.write(html)

    print(f"✓ Interactive graph saved to: {output_file}")
    print(f"  Nodes: {subgraph.number_of_nodes()}")
    print(f"  Edges: {subgraph.number_of_edges()}")

    # Try to display in Jupyter
    try:
        from IPython.display import IFrame

        return IFrame(src=output_file, width="100%", height=750)
    except (ImportError, NameError):
        print(f"  Open {output_file} in your browser to view")


# Update old function to use the new one
def visualize_subgraph_interactive(
    query, G, search_func, search_params, k=2, hops=1, output_file="graph.html"
):
    """Get subgraph and visualize it."""
    subgraph, results = get_subgraph_around_query(
        query, G, search_func, search_params, k, hops
    )
    return visualize_graph_with_pyvis(subgraph, output_file)
