from turingdb_kgsearch.subgraph import get_subgraph_around_query


def visualize_graph_with_pyvis(
    subgraph,
    output_file="graph.html",
    color_map=None,
    node_label_func=None,
    node_size_func=None,
    node_hover_func=None,
    default_node_color="#95a5a6",
    score_attributes=None,
):
    """
    Visualize a NetworkX graph using PyVis (generic version).

    Args:
        subgraph: NetworkX graph to visualize
        output_file: HTML file to save
        color_map: Dict of {node_type: color} or None for auto-generation
        node_label_func: Function(node_id, node_data) -> label string
        node_size_func: Function(node_id, node_data) -> size (int)
        node_hover_func: Function(node_id, node_data) -> hover text (HTML string)
        default_node_color: Fallback color for unknown node types
        score_attributes: List of attribute names to check for scoring
                         (e.g., ['relevance', 'combined_score', 'score'])

    Returns:
        IFrame for Jupyter display (if available)
    """
    from pyvis.network import Network

    # Default score attributes to look for
    if score_attributes is None:
        score_attributes = ["relevance", "combined_score", "score", "importance"]

    # Auto-generate color map if not provided
    if color_map is None:
        # Get unique node types
        node_types = set()
        for node, data in subgraph.nodes(data=True):
            node_type = data.get("type", "default")
            node_types.add(node_type)

        # Generate distinct colors
        color_palette = [
            "#ff6b6b",
            "#4ecdc4",
            "#95e1d3",
            "#ffe66d",
            "#a8e6cf",
            "#ffd3b6",
            "#ffaaa5",
            "#ff8b94",
            "#c7ceea",
            "#b4f8c8",
        ]

        color_map = {}
        for i, node_type in enumerate(sorted(node_types)):
            color_map[node_type] = color_palette[i % len(color_palette)]

        print(f"Auto-generated color map for {len(node_types)} node types:")
        for nt, color in color_map.items():
            print(f"  {nt}: {color}")

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
        node_type = node_data.get("type", "default")

        # Color
        color = color_map.get(node_type, default_node_color)

        # Find score for sizing (check multiple possible attributes)
        score = 0
        for attr in score_attributes:
            if attr in node_data:
                score = node_data[attr]
                break

        # Size
        if node_size_func:
            size = node_size_func(node, node_data)
        else:
            # Default: size based on score, with minimum size
            if score > 0:
                size = 30 + (score * 50)
            else:
                size = 20

        border_width = 3 if score > 0 else 1

        # Label
        if node_label_func:
            label = node_label_func(node, node_data)
        else:
            # Default label: try common attributes
            label = _default_label(node, node_data)

        # Hover text
        if node_hover_func:
            title = node_hover_func(node, node_data)
        else:
            title = _default_hover_text(node, node_data, score_attributes)

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
            borderWidthSelected=5,
            font={"size": 12 if score > 0 else 10},
        )

    # Add edges
    for source, target in subgraph.edges():
        edge_data = subgraph.get_edge_data(source, target)

        # Try to find relationship label
        relationship = ""
        if edge_data:
            for attr in ["rel", "relationship", "type", "label"]:
                if attr in edge_data:
                    relationship = edge_data[attr]
                    break

        net.add_edge(
            source, target, title=relationship, arrows="to", color="#888888", width=1
        )

    # Generate legend
    legend_html = _generate_legend(color_map, score_attributes)

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


def _default_label(node, node_data):
    """Generate default label for a node."""
    node_type = node_data.get("type", "unknown")

    # Try common label attributes
    for attr in ["name", "label", "title"]:
        if attr in node_data:
            text = str(node_data[attr])[:30]
            return f"{text}"

    # Fallback: use node_type + truncated node_id
    node_str = str(node)
    if len(node_str) > 20:
        node_str = node_str[:17] + "..."
    return f"{node_type}\n{node_str}"


def _default_hover_text(node, node_data, score_attributes):
    """Generate default hover text for a node."""
    node_type = node_data.get("type", "unknown")
    title_parts = [f"<b>{node_type.title()}</b>"]
    title_parts.append(f"ID: {node}")

    # Add main content attributes
    content_attrs = ["statement", "content", "description", "name", "text", "label"]
    for attr in content_attrs:
        if attr in node_data:
            text = str(node_data[attr])[:200]
            title_parts.append(f"<i>{attr}:</i> {text}...")
            break

    # Add scores
    for attr in score_attributes:
        if attr in node_data:
            value = node_data[attr]
            title_parts.append(f"<b>{attr.title()}: {value:.3f}</b>")

    # Add special markers
    if node_data.get("is_seed"):
        title_parts.append("<b>🎯 SEED NODE</b>")

    # Add other numeric attributes
    for key, value in node_data.items():
        if key not in ["type"] + score_attributes + content_attrs:
            if isinstance(value, (int, float)):
                title_parts.append(f"{key}: {value:.3f}")

    return "<br>".join(title_parts)


def _generate_legend(color_map, score_attributes):
    """Generate HTML legend for the visualization."""
    legend_items = []
    for node_type, color in sorted(color_map.items()):
        legend_items.append(
            f"<span style='color: {color};'>● {node_type.title()}</span><br>"
        )

    score_label = score_attributes[0] if score_attributes else "Score"

    legend_html = f"""
    <div style='position: absolute; top: 10px; right: 10px;
                background: white; padding: 10px; border: 1px solid #ccc;
                border-radius: 5px; font-family: Arial; font-size: 12px;'>
        <b>Node Types:</b><br>
        {''.join(legend_items)}
        <br><b>Node Size:</b> {score_label.title()}
    </div>
    """
    return legend_html


def extract_and_visualize_subgraph(
    query, G, search_func, search_params, k=2, hops=1, output_file="graph.html"
):
    """Get subgraph and visualize it."""
    subgraph, results = get_subgraph_around_query(
        query, G, search_func, search_params, k, hops
    )
    return visualize_graph_with_pyvis(subgraph, output_file)
