def graph_to_llm_context(
    G, format="markdown", include_attributes=True, max_nodes=None, node_label_key="type"
):
    """
    Convert a NetworkX graph to LLM-friendly text format.

    Args:
        G: NetworkX graph
        format: Output format - 'markdown', 'json', 'natural', or 'cypher'
        include_attributes: Whether to include node/edge attributes
        max_nodes: Maximum nodes to include (None = all)
        node_label_key: Which node attribute to use as label (e.g., 'type', 'name')

    Returns:
        String representation suitable for LLM context
    """

    if format == "markdown":
        return _graph_to_markdown(G, include_attributes, max_nodes, node_label_key)
    elif format == "json":
        return _graph_to_json(G, include_attributes, max_nodes)
    elif format == "natural":
        return _graph_to_natural_language(
            G, include_attributes, max_nodes, node_label_key
        )
    elif format == "cypher":
        return _graph_to_cypher(G, include_attributes, max_nodes)
    else:
        raise ValueError(f"Unknown format: {format}")


def _graph_to_markdown(G, include_attributes, max_nodes, node_label_key):
    """Convert graph to markdown format."""
    lines = []

    # Header
    lines.append("# Graph Structure\n")
    lines.append(f"**Nodes:** {G.number_of_nodes()}")
    lines.append(f"**Edges:** {G.number_of_edges()}\n")

    # Limit nodes if specified
    nodes = list(G.nodes())[:max_nodes] if max_nodes else list(G.nodes())

    # Nodes section
    lines.append("## Nodes\n")
    for node in nodes:
        node_data = G.nodes[node]

        lines.append(f"### Node: `{node}`")
        if include_attributes and node_data:
            for key, value in node_data.items():
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200] + "..."
                lines.append(f"- **{key}:** {str_value}")
        lines.append("")

    # Edges section
    lines.append("## Edges\n")
    for source, target, data in G.edges(data=True):
        if max_nodes and (source not in nodes or target not in nodes):
            continue

        edge_desc = f"`{source}` → `{target}`"

        if include_attributes and data:
            lines.append(f"### {edge_desc}")
            for key, value in data.items():
                lines.append(f"- **{key}:** {value}")
        else:
            lines.append(f"- {edge_desc}")
        lines.append("")

    return "\n".join(lines)


def _graph_to_json(G, include_attributes, max_nodes):
    """Convert graph to JSON format."""
    import json

    nodes = list(G.nodes())[:max_nodes] if max_nodes else list(G.nodes())

    graph_dict = {
        "metadata": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "directed": G.is_directed(),
        },
        "nodes": [],
        "edges": [],
    }

    # Add nodes
    for node in nodes:
        node_dict = {"id": str(node)}
        if include_attributes:
            node_dict["attributes"] = {
                k: str(v)[:200] for k, v in G.nodes[node].items()
            }
        graph_dict["nodes"].append(node_dict)

    # Add edges
    for source, target, data in G.edges(data=True):
        if max_nodes and (source not in nodes or target not in nodes):
            continue

        edge_dict = {"source": str(source), "target": str(target)}
        if include_attributes and data:
            edge_dict["attributes"] = {k: str(v) for k, v in data.items()}
        graph_dict["edges"].append(edge_dict)

    return json.dumps(graph_dict, indent=2)


def _graph_to_natural_language(G, include_attributes, max_nodes, node_label_key):
    """Convert graph to natural language description."""
    lines = []

    nodes = list(G.nodes())[:max_nodes] if max_nodes else list(G.nodes())

    # Summary
    lines.append(
        f"This graph contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )
    lines.append("")

    # Node types breakdown
    node_types = {}
    for node in nodes:
        node_type = G.nodes[node].get(node_label_key, "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    if node_types:
        lines.append("Node types:")
        for ntype, count in node_types.items():
            lines.append(f"- {count} {ntype} node(s)")
        lines.append("")

    # Describe nodes
    lines.append("Nodes in the graph:")
    for node in nodes:
        node_data = G.nodes[node]
        node_type = node_data.get(node_label_key, "node")

        desc = f"- {node} is a {node_type}"

        if include_attributes:
            # Add key attributes
            key_attrs = []
            for key in ["name", "title", "label", "statement"]:
                if key in node_data:
                    value = str(node_data[key])[:100]
                    key_attrs.append(f"{key}: '{value}'")

            if key_attrs:
                desc += f" ({', '.join(key_attrs)})"

        lines.append(desc)

    lines.append("")

    # Describe relationships
    lines.append("Relationships:")
    for source, target, data in G.edges(data=True):
        if max_nodes and (source not in nodes or target not in nodes):
            continue

        rel_type = data.get("rel", data.get("relationship", "connected to"))
        lines.append(f"- {source} {rel_type} {target}")

    return "\n".join(lines)


def _graph_to_cypher(G, include_attributes, max_nodes):
    """Convert graph to Cypher queries (Neo4j style)."""
    lines = []

    nodes = list(G.nodes())[:max_nodes] if max_nodes else list(G.nodes())

    lines.append("// Create nodes")
    for node in nodes:
        node_data = G.nodes[node]
        node_label = node_data.get("type", "Node")

        if include_attributes and node_data:
            props = []
            for key, value in node_data.items():
                if key != "type":
                    # Escape strings
                    str_value = str(value).replace("'", "\\'")[:200]
                    props.append(f"{key}: '{str_value}'")

            props_str = ", ".join(props)
            lines.append(f"CREATE (n_{node}:{node_label} {{{props_str}}})")
        else:
            lines.append(f"CREATE (n_{node}:{node_label})")

    lines.append("")
    lines.append("// Create relationships")
    for source, target, data in G.edges(data=True):
        if max_nodes and (source not in nodes or target not in nodes):
            continue

        rel_type = data.get("rel", data.get("relationship", "CONNECTED_TO")).upper()

        if include_attributes and data:
            props = []
            for key, value in data.items():
                if key not in ["rel", "relationship"]:
                    str_value = str(value).replace("'", "\\'")
                    props.append(f"{key}: '{str_value}'")

            props_str = "{" + ", ".join(props) + "}" if props else ""
            lines.append(f"CREATE (n_{source})-[:{rel_type} {props_str}]->(n_{target})")
        else:
            lines.append(f"CREATE (n_{source})-[:{rel_type}]->(n_{target})")

    return "\n".join(lines)


def create_llm_prompt_with_graph(
    query, subgraph, report, format="natural", custom_task=None
):
    """
    Create a complete LLM prompt with graph context.

    Args:
        query: Original user query
        subgraph: NetworkX subgraph
        report: Text report from print_hybrid_workflow_results
        format: Graph serialization format
        custom_task: Optional custom task description (overrides default)

    Returns:
        Complete prompt string for LLM
    """

    graph_context = graph_to_llm_context(
        subgraph, format=format, include_attributes=True, max_nodes=50
    )

    if custom_task is None:
        task = """## Task
Based ONLY on the search results and graph structure provided above (ignore any external knowledge):

1. Answer the user's query using only the information present in the graph
2. Identify key nodes and relationships relevant to the query
3. Explain how different nodes are connected and what patterns emerge
4. Highlight any important information or insights from the subgraph

IMPORTANT: Base your response exclusively on the provided graph data. Do not use external knowledge."""
    else:
        task = f"## Task\n{custom_task}"

    prompt = f"""# Graph-Based Query Response

## User Query
"{query}"

## Search Results
{report}

## Graph Structure
{graph_context}

{task}
"""

    return prompt


def query_llm(
    prompt,
    system_prompt=None,
    provider="OpenAI",
    model=None,
    api_key=None,
    temperature=0.0,
):
    """Simple LLM query function with optional system prompt"""

    if provider == "OpenAI":
        import openai

        client = openai.OpenAI(api_key=api_key)
        model = model or "gpt-4o-mini"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content

    elif provider == "Mistral":
        import mistralai

        client = mistralai.Mistral(api_key=api_key)
        model = model or "mistral-small-latest"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.complete(
            model=model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content

    elif provider == "Anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        model = model or "claude-3-5-haiku-latest"

        response = client.messages.create(
            model=model,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=temperature,
        )
        return response.content[0].text

    else:
        raise ValueError(f"Unsupported provider: {provider}")
