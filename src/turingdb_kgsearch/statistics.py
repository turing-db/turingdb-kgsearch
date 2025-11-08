import networkx as nx


def get_subgraph_stats(
    subgraph, include_node_breakdown=True, include_centrality=True, include_paths=True
):
    """
    Comprehensive statistics about a subgraph.

    Args:
        subgraph: NetworkX graph
        include_node_breakdown: Include node type distribution
        include_centrality: Calculate centrality measures
        include_paths: Analyze path metrics

    Returns:
        Dictionary with comprehensive graph statistics
    """
    stats = {}

    # Basic counts
    stats["basic"] = {
        "nodes": subgraph.number_of_nodes(),
        "edges": subgraph.number_of_edges(),
        "density": nx.density(subgraph),
        "is_connected": nx.is_weakly_connected(subgraph)
        if subgraph.is_directed()
        else nx.is_connected(subgraph),
    }

    # Degree statistics
    degrees = [d for n, d in subgraph.degree()]
    if degrees:
        stats["degree"] = {
            "average": sum(degrees) / len(degrees),
            "max": max(degrees),
            "min": min(degrees),
            "median": sorted(degrees)[len(degrees) // 2] if degrees else 0,
        }

    # Node breakdown by type/attributes
    if include_node_breakdown:
        # By type
        node_types = {}
        for node in subgraph.nodes():
            ntype = subgraph.nodes[node].get("type", "unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1
        stats["node_types"] = node_types

        # Seed vs found vs intermediate
        seed_count = sum(
            1 for n in subgraph.nodes() if subgraph.nodes[n].get("is_seed", False)
        )
        intermediate_count = sum(
            1
            for n in subgraph.nodes()
            if subgraph.nodes[n].get("is_intermediate", False)
        )
        found_count = subgraph.number_of_nodes() - seed_count - intermediate_count

        stats["node_roles"] = {
            "seed": seed_count,
            "found": found_count,
            "intermediate": intermediate_count,
        }

    # Centrality measures
    if include_centrality and subgraph.number_of_nodes() > 0:
        try:
            # Degree centrality
            degree_cent = nx.degree_centrality(subgraph)
            stats["centrality"] = {
                "top_by_degree": sorted(
                    degree_cent.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }

            # Betweenness (expensive for large graphs)
            if subgraph.number_of_nodes() < 100:
                between_cent = nx.betweenness_centrality(subgraph)
                stats["centrality"]["top_by_betweenness"] = sorted(
                    between_cent.items(), key=lambda x: x[1], reverse=True
                )[:5]

            # Closeness
            if subgraph.number_of_nodes() < 100:
                if (
                    nx.is_weakly_connected(subgraph)
                    if subgraph.is_directed()
                    else nx.is_connected(subgraph)
                ):
                    close_cent = nx.closeness_centrality(subgraph)
                    stats["centrality"]["top_by_closeness"] = sorted(
                        close_cent.items(), key=lambda x: x[1], reverse=True
                    )[:5]
        except (nx.NetworkXError, ZeroDivisionError, ValueError):
            stats["centrality"] = {"error": "Could not compute centrality"}

    # Path analysis
    if include_paths and subgraph.number_of_nodes() > 1:
        seed_nodes = [
            n for n in subgraph.nodes() if subgraph.nodes[n].get("is_seed", False)
        ]

        if len(seed_nodes) >= 2:
            path_lengths = []
            for i, n1 in enumerate(seed_nodes):
                for n2 in seed_nodes[i + 1 :]:
                    try:
                        if subgraph.is_directed():
                            length = nx.shortest_path_length(subgraph, n1, n2)
                        else:
                            length = nx.shortest_path_length(subgraph, n1, n2)
                        path_lengths.append(length)
                    except nx.NetworkXNoPath:
                        pass

            if path_lengths:
                stats["paths"] = {
                    "avg_distance_between_seeds": sum(path_lengths) / len(path_lengths),
                    "max_distance": max(path_lengths),
                    "min_distance": min(path_lengths),
                }

        # Diameter (expensive)
        if subgraph.number_of_nodes() < 50:
            try:
                if (
                    nx.is_weakly_connected(subgraph)
                    if subgraph.is_directed()
                    else nx.is_connected(subgraph)
                ):
                    stats["diameter"] = nx.diameter(subgraph)
                    stats["radius"] = nx.radius(subgraph)
            except (nx.NetworkXError, nx.NetworkXNotImplemented):
                pass

    # Clustering (for undirected or convert to undirected)
    if not subgraph.is_directed() and subgraph.number_of_nodes() > 0:
        try:
            stats["clustering"] = {
                "average": nx.average_clustering(subgraph),
                "transitivity": nx.transitivity(subgraph),
            }
        except (nx.NetworkXError, ZeroDivisionError):
            pass

    # Edge type distribution (if edges have 'rel' or 'relationship' attribute)
    edge_types = {}
    for u, v, data in subgraph.edges(data=True):
        etype = data.get("rel", data.get("relationship", "unknown"))
        edge_types[etype] = edge_types.get(etype, 0) + 1

    if edge_types:
        stats["edge_types"] = edge_types

    # Relevance statistics (if similarity scores present)
    relevance_scores = []
    for node in subgraph.nodes():
        score = subgraph.nodes[node].get(
            "combined_score", subgraph.nodes[node].get("relevance", None)
        )
        if score is not None:
            relevance_scores.append(score)

    if relevance_scores:
        stats["relevance"] = {
            "avg_score": sum(relevance_scores) / len(relevance_scores),
            "max_score": max(relevance_scores),
            "min_score": min(relevance_scores),
            "nodes_with_scores": len(relevance_scores),
        }

    return stats


def print_subgraph_stats(stats, verbose=True):
    """Pretty print subgraph statistics."""

    print("\n" + "=" * 80)
    print("SUBGRAPH STATISTICS")
    print("=" * 80)

    # Basic
    if "basic" in stats:
        print("\n📊 Basic Metrics:")
        for key, value in stats["basic"].items():
            print(f"   {key}: {value}")

    # Degree
    if "degree" in stats:
        print("\n🔗 Degree Statistics:")
        for key, value in stats["degree"].items():
            print(f"   {key}: {value:.2f}")

    # Node types
    if "node_types" in stats:
        print("\n🏷️  Node Types:")
        for ntype, count in stats["node_types"].items():
            print(f"   {ntype}: {count}")

    # Node roles
    if "node_roles" in stats:
        print("\n🎯 Node Roles:")
        for role, count in stats["node_roles"].items():
            print(f"   {role}: {count}")

    # Centrality
    if "centrality" in stats and verbose:
        print("\n⭐ Most Central Nodes:")
        if "top_by_degree" in stats["centrality"]:
            print("   By Degree:")
            for node, score in stats["centrality"]["top_by_degree"]:
                print(f"      {node}: {score:.3f}")

        if "top_by_betweenness" in stats["centrality"]:
            print("   By Betweenness:")
            for node, score in stats["centrality"]["top_by_betweenness"]:
                print(f"      {node}: {score:.3f}")

    # Paths
    if "paths" in stats:
        print("\n🛤️  Path Metrics:")
        for key, value in stats["paths"].items():
            print(f"   {key}: {value:.2f}")

    if "diameter" in stats:
        print(f"   diameter: {stats['diameter']}")
        print(f"   radius: {stats['radius']}")

    # Clustering
    if "clustering" in stats:
        print("\n🔄 Clustering:")
        for key, value in stats["clustering"].items():
            print(f"   {key}: {value:.3f}")

    # Edge types
    if "edge_types" in stats:
        print("\n🔗 Edge Types:")
        for etype, count in stats["edge_types"].items():
            print(f"   {etype}: {count}")

    # Relevance
    if "relevance" in stats:
        print("\n📈 Relevance Scores:")
        for key, value in stats["relevance"].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

    print("\n" + "=" * 80)
