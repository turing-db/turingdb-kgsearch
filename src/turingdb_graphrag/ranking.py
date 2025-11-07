import networkx as nx


def rank_nodes_by_importance(subgraph, methods="all", top_k=10, aggregate="average"):
    """
    Rank nodes by importance using multiple centrality measures.

    Args:
        subgraph: NetworkX graph
        methods: 'all' or list of methods ['pagerank', 'degree', 'betweenness',
                 'closeness', 'eigenvector', 'hits', 'katz']
        top_k: Number of top nodes to return
        aggregate: How to combine scores - 'average', 'max', 'weighted', or dict of weights

    Returns:
        Dictionary with rankings by each method + combined ranking
    """

    sub = subgraph
    nodes_to_rank = list(sub.nodes())

    if len(nodes_to_rank) == 0:
        return {"error": "No nodes to rank"}

    # Determine which methods to use
    if methods == "all":
        methods_to_use = [
            "pagerank",
            "degree",
            "betweenness",
            "closeness",
            "eigenvector",
            "relevance",
        ]
    else:
        methods_to_use = methods if isinstance(methods, list) else [methods]

    results = {}
    scores_by_method = {}  # Store raw scores for aggregation

    # 1. PageRank (importance via incoming links)
    if "pagerank" in methods_to_use:
        try:
            pagerank = nx.pagerank(sub, alpha=0.85)
            results["pagerank"] = sorted(
                pagerank.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            scores_by_method["pagerank"] = pagerank
        except Exception as e:
            results["pagerank"] = {"error": str(e)}

    # 2. Degree Centrality (number of connections)
    if "degree" in methods_to_use:
        try:
            degree = nx.degree_centrality(sub)
            results["degree"] = sorted(
                degree.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            scores_by_method["degree"] = degree
        except Exception as e:
            results["degree"] = {"error": str(e)}

    # 3. Betweenness Centrality (bridge nodes) - expensive
    if "betweenness" in methods_to_use and len(nodes_to_rank) < 200:
        try:
            betweenness = nx.betweenness_centrality(sub)
            results["betweenness"] = sorted(
                betweenness.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            scores_by_method["betweenness"] = betweenness
        except Exception as e:
            results["betweenness"] = {"error": str(e)}

    # 4. Closeness Centrality (average distance to others)
    if "closeness" in methods_to_use and len(nodes_to_rank) < 200:
        try:
            # Only for connected graphs
            if (
                nx.is_weakly_connected(sub)
                if sub.is_directed()
                else nx.is_connected(sub)
            ):
                closeness = nx.closeness_centrality(sub)
                results["closeness"] = sorted(
                    closeness.items(), key=lambda x: x[1], reverse=True
                )[:top_k]
                scores_by_method["closeness"] = closeness
            else:
                results["closeness"] = {
                    "error": "graph is not connected enough to compute closeness centrality"
                }
        except Exception as e:
            results["closeness"] = {"error": str(e)}

    # 5. Eigenvector Centrality (connected to important nodes)
    if "eigenvector" in methods_to_use and len(nodes_to_rank) < 500:
        try:
            eigenvector = nx.eigenvector_centrality(sub, max_iter=1000)
            results["eigenvector"] = sorted(
                eigenvector.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            scores_by_method["eigenvector"] = eigenvector
        except Exception as e:
            results["eigenvector"] = {"error": str(e)}

    # 6. Relevance Score (from search results)
    if "relevance" in methods_to_use:
        relevance = {}
        for node in nodes_to_rank:
            score = sub.nodes[node].get(
                "combined_score",
                sub.nodes[node].get("relevance", sub.nodes[node].get("similarity", 0)),
            )
            if score > 0:
                relevance[node] = score

        if relevance:
            results["relevance"] = sorted(
                relevance.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            scores_by_method["relevance"] = relevance

    # 7. Aggregate ranking (combine all methods)
    if len(scores_by_method) > 1:
        combined = _aggregate_rankings(
            scores_by_method, nodes_to_rank, aggregate, top_k
        )
        results["combined"] = combined

    # Add metadata
    results["metadata"] = {
        "total_nodes": len(nodes_to_rank),
        "methods_used": list(scores_by_method.keys()),
        "aggregation": aggregate,
    }

    return results


def _aggregate_rankings(scores_by_method, nodes, method="average", top_k=10):
    """
    Aggregate multiple ranking methods into a single ranking.

    Args:
        scores_by_method: Dict of {method_name: {node: score}}
        nodes: List of nodes to rank
        method: 'average', 'max', 'weighted', or dict of weights
        top_k: Number to return

    Returns:
        List of (node, combined_score) tuples
    """

    # Normalize all scores to [0, 1]
    normalized = {}
    for method_name, scores in scores_by_method.items():
        if not scores:
            continue

        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        range_score = max_score - min_score if max_score != min_score else 1

        normalized[method_name] = {
            node: (score - min_score) / range_score for node, score in scores.items()
        }

    # Determine weights
    if method == "average":
        weights = {m: 1.0 / len(normalized) for m in normalized.keys()}
    elif method == "max":
        weights = None  # Will take max instead
    elif isinstance(method, dict):
        weights = method
    else:
        weights = {m: 1.0 / len(normalized) for m in normalized.keys()}

    # Combine scores
    combined_scores = {}
    for node in nodes:
        if method == "max":
            # Take maximum score across methods
            node_scores = [normalized[m].get(node, 0) for m in normalized.keys()]
            combined_scores[node] = max(node_scores) if node_scores else 0
        else:
            # Weighted average
            score = 0
            total_weight = 0
            for method_name, norm_scores in normalized.items():
                if node in norm_scores:
                    weight = weights.get(method_name, 0)
                    score += norm_scores[node] * weight
                    total_weight += weight

            combined_scores[node] = score / total_weight if total_weight > 0 else 0

    # Sort and return top k
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def rank_nodes_by_importance_with_context(
    subgraph, focus_type="control", methods="all", top_k=10, aggregate="average"
):
    """
    Rank ALL nodes but highlight nodes of focus_type.
    Preserves graph structure for accurate centrality.
    """

    # Rank all nodes (preserves connections)
    rankings = rank_nodes_by_importance(
        subgraph,
        methods=methods,
        top_k=top_k,  # Get more to ensure we have enough of focus_type
        aggregate=aggregate,
    )

    # Filter results to focus_type
    filtered_rankings = {}
    for method, nodes in rankings.items():
        if method == "metadata":
            filtered_rankings[method] = rankings[method]
            continue

        if isinstance(nodes, dict):
            filtered_rankings[method] = nodes
            continue

        # Keep only focus_type nodes
        filtered_rankings[method] = [
            (n, s) for n, s in nodes if subgraph.nodes[n].get("type") == focus_type
        ][:top_k]

    return filtered_rankings, rankings  # Return both


def print_node_rankings(rankings, subgraph=None, show_details=True):
    """Pretty print node rankings."""

    print("\n" + "=" * 80)
    print("NODE IMPORTANCE RANKINGS")
    print("=" * 80)

    if "metadata" in rankings:
        meta = rankings["metadata"]
        print(
            f"\nRanked {meta['total_nodes']} nodes using: {', '.join(meta['methods_used'])}"
        )
        print()

    # Print each method's top nodes
    for method, ranked_nodes in rankings.items():
        if method in ["metadata", "combined"]:
            continue

        if isinstance(ranked_nodes, dict) and "error" in ranked_nodes:
            print(f"\n❌ {method.upper()}: {ranked_nodes['error']}")
            continue

        print(f"\n📊 {method.upper()} (Top 5):")
        for i, (node, score) in enumerate(ranked_nodes[:5], 1):
            node_str = str(node)
            if subgraph and show_details:
                node_type = subgraph.nodes[node].get("type", "unknown")
                node_str = f"{node} ({node_type})"
            print(f"   {i}. {node_str}: {score:.4f}")

    # Print combined ranking
    if "combined" in rankings:
        print("\n⭐ COMBINED RANKING (Top 10):")
        for i, (node, score) in enumerate(rankings["combined"][:10], 1):
            node_str = str(node)
            if subgraph and show_details:
                node_type = subgraph.nodes[node].get("type", "unknown")
                node_str = f"{node} ({node_type})"
            print(f"   {i}. {node_str}: {score:.4f}")

    print("\n" + "=" * 80)


def compare_node_importance(node, rankings):
    """
    Show how a specific node ranks across different methods.

    Args:
        node: Node ID to analyze
        rankings: Output from rank_nodes_by_importance

    Returns:
        Dictionary of rankings for this node
    """

    node_rankings = {}

    for method, ranked_nodes in rankings.items():
        if method in ["metadata", "combined"]:
            continue

        if isinstance(ranked_nodes, dict) and "error" in ranked_nodes:
            continue

        # Find position and score
        for i, (n, score) in enumerate(ranked_nodes, 1):
            if n == node:
                node_rankings[method] = {
                    "rank": i,
                    "score": score,
                    "percentile": (1 - i / len(ranked_nodes)) * 100,
                }
                break

    return node_rankings


def diagnose_rankings(subgraph, node_type="control"):
    """Debug why rankings might be zero."""

    # Check filtering
    all_nodes = list(subgraph.nodes())
    filtered_nodes = [
        n for n in all_nodes if subgraph.nodes[n].get("type") == node_type
    ]

    print(f"Total nodes: {len(all_nodes)}")
    print(f"Filtered nodes ({node_type}): {len(filtered_nodes)}")

    # Check connectivity
    filtered_sub = subgraph.subgraph(filtered_nodes)
    if filtered_sub.is_directed():
        connected = nx.is_weakly_connected(filtered_sub)
    else:
        connected = nx.is_connected(filtered_sub)

    print(f"Filtered subgraph connected: {connected}")
    print(f"Filtered subgraph edges: {filtered_sub.number_of_edges()}")

    # Check if isolated
    isolated = list(nx.isolates(filtered_sub))
    print(f"Isolated nodes: {len(isolated)}/{len(filtered_nodes)}")

    if len(isolated) == len(filtered_nodes):
        print("\n⚠️  WARNING: All filtered nodes are isolated!")
        print("   Recommendation: Don't filter by node_type for ranking.")
        print("   Rank on full graph, then filter results for display.")
