import networkx as nx


def get_subgraph_around_query(query, G, search_func, search_params, k=3, hops=1):
    """
    Search and get a subgraph around relevant nodes.

    Args:
        query: Search query
        G: NetworkX graph
        search_func: Search function to use (dense_search, sparse_search, or hybrid_search)
        search_params: Dictionary of parameters for the search function
        k: Number of seed nodes
        hops: How many edges to expand

    Returns:
        Tuple of (subgraph, results)
    """

    # Search using the specified function
    results = search_func(query=query, G=G, k=k, **search_params)
    seed_nodes = [r["node_id"] for r in results]

    # Expand to neighbors
    expanded = set(seed_nodes)
    for node in seed_nodes:
        try:
            # Predecessors
            for n in nx.single_source_shortest_path_length(
                G.reverse(), node, cutoff=hops
            ).keys():
                expanded.add(n)
            # Successors
            for n in nx.single_source_shortest_path_length(G, node, cutoff=hops).keys():
                expanded.add(n)
        except KeyError:
            pass

    # Create subgraph
    subgraph = G.subgraph(expanded).copy()

    # Add relevance scores
    for r in results:
        if r["node_id"] in subgraph:
            subgraph.nodes[r["node_id"]]["relevance"] = r["similarity"]

    return subgraph, results
