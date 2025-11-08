import networkx as nx
import numpy as np
from turingdb_kgsearch.search import hybrid_search


def get_hybrid_filtered_neighborhood(
    seed_node,
    query,
    G,
    structural_vectors,
    node_vectors,
    node_texts,
    model,
    max_hops=2,
    min_structural_sim=0.7,
    min_semantic_sim=0.6,
    structural_weight=0.5,
):
    """
    Get neighborhood from graph, filtered by BOTH structural AND semantic similarity.

    Args:
        seed_node: Starting node
        query: Original search query (for semantic filtering)
        G: NetworkX graph
        structural_vectors: Node2Vec embeddings
        node_vectors: Dense (text) embeddings
        node_texts: Node text dictionary
        model: SentenceTransformer model
        max_hops: Maximum graph distance
        min_structural_sim: Minimum Node2Vec similarity threshold
        min_semantic_sim: Minimum semantic similarity to query
        structural_weight: Weight for combining scores (0.5 = balanced)

    Returns:
        List of neighbors with combined scores
    """
    if seed_node not in structural_vectors:
        return []

    seed_struct_vec = structural_vectors[seed_node]

    # Encode query for semantic filtering
    query_semantic_vec = model.encode([query])[0]

    neighbors = []

    # Get all nodes within max_hops
    for hop in range(1, max_hops + 1):
        # Predecessors
        try:
            pred_nodes = nx.single_source_shortest_path_length(
                G.reverse(), seed_node, cutoff=hop
            )
            for node, distance in pred_nodes.items():
                if node == seed_node or distance != hop:
                    continue

                # Check both structural AND semantic similarity
                if node in structural_vectors and node in node_vectors:
                    # 1. Structural similarity (Node2Vec)
                    node_struct_vec = structural_vectors[node]
                    structural_sim = np.dot(seed_struct_vec, node_struct_vec) / (
                        np.linalg.norm(seed_struct_vec)
                        * np.linalg.norm(node_struct_vec)
                    )

                    # 2. Semantic similarity (to original query)
                    node_semantic_vec = node_vectors[node]
                    semantic_sim = np.dot(query_semantic_vec, node_semantic_vec) / (
                        np.linalg.norm(query_semantic_vec)
                        * np.linalg.norm(node_semantic_vec)
                    )

                    # 3. Combined filtering
                    if (
                        structural_sim >= min_structural_sim
                        and semantic_sim >= min_semantic_sim
                    ):
                        # Hybrid score
                        combined_score = (
                            structural_weight * structural_sim
                            + (1 - structural_weight) * semantic_sim
                        )

                        neighbors.append(
                            {
                                "node_id": node,
                                "hop_distance": distance,
                                "direction": "predecessor",
                                "structural_similarity": float(structural_sim),
                                "semantic_similarity": float(semantic_sim),
                                "combined_score": float(combined_score),
                            }
                        )
        except (nx.NetworkXError, KeyError):
            pass

        # Successors (same logic)
        try:
            succ_nodes = nx.single_source_shortest_path_length(G, seed_node, cutoff=hop)
            for node, distance in succ_nodes.items():
                if node == seed_node or distance != hop:
                    continue

                if node in structural_vectors and node in node_vectors:
                    # Structural similarity
                    node_struct_vec = structural_vectors[node]
                    structural_sim = np.dot(seed_struct_vec, node_struct_vec) / (
                        np.linalg.norm(seed_struct_vec)
                        * np.linalg.norm(node_struct_vec)
                    )

                    # Semantic similarity
                    node_semantic_vec = node_vectors[node]
                    semantic_sim = np.dot(query_semantic_vec, node_semantic_vec) / (
                        np.linalg.norm(query_semantic_vec)
                        * np.linalg.norm(node_semantic_vec)
                    )

                    # Combined filtering
                    if (
                        structural_sim >= min_structural_sim
                        and semantic_sim >= min_semantic_sim
                    ):
                        combined_score = (
                            structural_weight * structural_sim
                            + (1 - structural_weight) * semantic_sim
                        )

                        neighbors.append(
                            {
                                "node_id": node,
                                "hop_distance": distance,
                                "direction": "successor",
                                "structural_similarity": float(structural_sim),
                                "semantic_similarity": float(semantic_sim),
                                "combined_score": float(combined_score),
                            }
                        )
        except (nx.NetworkXError, KeyError):
            pass

    # Remove duplicates and sort by combined score
    seen = set()
    unique_neighbors = []
    for n in neighbors:
        if n["node_id"] not in seen:
            seen.add(n["node_id"])
            unique_neighbors.append(n)

    unique_neighbors.sort(key=lambda x: x["combined_score"], reverse=True)
    return unique_neighbors


def search_and_expand_hybrid_filtered(
    query,
    G,
    node_vectors,
    node_texts,
    sparse_vectors,
    sparse_vectorizer,
    structural_vectors,
    model,
    k_search=3,
    max_hops=2,
    min_structural_sim=0.7,
    min_semantic_sim=0.6,
    structural_weight=0.5,
    alpha=0.7,
    node_type="control",
):
    """
    Two-stage search with HYBRID filtering (structural + semantic):
    1. Find semantically relevant nodes (hybrid search)
    2. Traverse graph, keeping neighbors that are BOTH structurally AND semantically similar

    Args:
        query: Search query
        G: NetworkX graph
        node_vectors: Dense vectors
        node_texts: Node texts
        sparse_vectors: Sparse vectors
        sparse_vectorizer: TF-IDF vectorizer
        structural_vectors: Node2Vec vectors
        model: SentenceTransformer model
        k_search: Number of semantic matches
        max_hops: Maximum graph distance
        min_structural_sim: Minimum Node2Vec similarity threshold
        min_semantic_sim: Minimum semantic similarity to query threshold
        structural_weight: Weight for combining scores (0.5 = balanced)
        alpha: Hybrid search weight
        node_type: Filter by node type

    Returns:
        Tuple of (semantic_results, expanded_results, subgraph)
        - semantic_results: List of seed nodes from hybrid search
        - expanded_results: List of dicts with seed and neighbors info
        - subgraph: NetworkX subgraph with similarity scores added to nodes
    """

    # Stage 1: Semantic + keyword search
    print(f"Stage 1: Hybrid search for '{query}'...")
    print("-" * 80)

    semantic_results = hybrid_search(
        query=query,
        node_vectors=node_vectors,
        node_texts=node_texts,
        G=G,
        sparse_vectors=sparse_vectors,
        sparse_vectorizer=sparse_vectorizer,
        model=model,
        k=k_search,
        alpha=alpha,
        node_type=node_type,
    )

    print(f"\nFound {len(semantic_results)} semantically relevant seed nodes:")
    for i, r in enumerate(semantic_results, 1):
        print(f"  {i}. {r['node_id']} (score: {r['similarity']:.3f})")
        print(f"     {r['text'][:100]}...")

    # Stage 2: Graph traversal with HYBRID filtering
    print(f"\n{'='*80}")
    print("Stage 2: Hybrid filtering (structural + semantic)...")
    print(f"  - Max hops: {max_hops}")
    print(f"  - Min structural similarity: {min_structural_sim}")
    print(f"  - Min semantic similarity: {min_semantic_sim}")
    print(f"  - Structural weight: {structural_weight}")
    print("-" * 80)

    expanded = []
    all_nodes = set()  # Collect all nodes for subgraph
    node_scores = {}  # Store similarity scores

    # Add seed nodes
    for result in semantic_results:
        all_nodes.add(result["node_id"])
        node_scores[result["node_id"]] = {
            "is_seed": True,
            "seed_score": result["similarity"],
            "dense_score": result.get("dense_score"),
            "sparse_score": result.get("sparse_score"),
        }

    for result in semantic_results:
        seed_node = result["node_id"]
        print(f"\n  Expanding from: {seed_node}")

        neighbors = get_hybrid_filtered_neighborhood(
            seed_node=seed_node,
            query=query,
            G=G,
            structural_vectors=structural_vectors,
            node_vectors=node_vectors,
            node_texts=node_texts,
            model=model,
            max_hops=max_hops,
            min_structural_sim=min_structural_sim,
            min_semantic_sim=min_semantic_sim,
            structural_weight=structural_weight,
        )

        print(f"  Found {len(neighbors)} neighbors (after hybrid filtering):")
        for n in neighbors[:5]:  # Show top 5
            print(f"    - {n['node_id']} ({n['direction']}, {n['hop_distance']} hops)")
            print(
                f"      Combined: {n['combined_score']:.3f} (Struct: {n['structural_similarity']:.3f}, Sem: {n['semantic_similarity']:.3f})"
            )

        # Add full node data
        for n in neighbors:
            n["text"] = node_texts.get(n["node_id"], "")
            n["data"] = dict(G.nodes[n["node_id"]])

            # Collect nodes for subgraph
            all_nodes.add(n["node_id"])

            # Store scores
            node_scores[n["node_id"]] = {
                "is_seed": False,
                "seed_node": seed_node,
                "hop_distance": n["hop_distance"],
                "direction": n["direction"],
                "structural_similarity": n["structural_similarity"],
                "semantic_similarity": n["semantic_similarity"],
                "combined_score": n["combined_score"],
            }

        expanded.append({"seed": result, "hybrid_neighbors": neighbors})

    # Create subgraph from collected nodes
    print(f"\n{'='*80}")
    print("Building subgraph...")

    # Get paths between all nodes to include intermediate nodes and edges
    nodes_with_paths = set(all_nodes)
    for node1 in all_nodes:
        for node2 in all_nodes:
            if node1 != node2:
                try:
                    # Get shortest path
                    path = nx.shortest_path(G, node1, node2)
                    nodes_with_paths.update(path)
                except nx.NetworkXNoPath:
                    pass

    # Create subgraph
    subgraph = G.subgraph(nodes_with_paths).copy()

    # Add similarity scores to nodes
    for node in subgraph.nodes():
        if node in node_scores:
            for key, value in node_scores[node].items():
                subgraph.nodes[node][key] = value
        else:
            # Intermediate node (not directly found, but in path)
            subgraph.nodes[node]["is_seed"] = False
            subgraph.nodes[node]["is_intermediate"] = True

    print("✓ Subgraph created:")
    print(f"  Total nodes: {subgraph.number_of_nodes()}")
    print(
        f"  - Seed nodes: {sum(1 for n in subgraph.nodes() if subgraph.nodes[n].get('is_seed', False))}"
    )
    print(f"  - Found neighbors: {len(all_nodes) - k_search}")
    print(f"  - Intermediate nodes: {subgraph.number_of_nodes() - len(all_nodes)}")
    print(f"  Total edges: {subgraph.number_of_edges()}")

    return semantic_results, expanded, subgraph


def generate_report_hybrid_workflow_results(semantic_results, expanded, subgraph=None):
    """
    Generate hybrid-filtered workflow results report as a string.

    Args:
        semantic_results: List of seed nodes from hybrid search
        expanded: List of dicts with seed and neighbors info
        subgraph: NetworkX subgraph (optional)

    Returns:
        String containing the full report
    """
    lines = []

    lines.append("=" * 80)
    lines.append("HYBRID-FILTERED WORKFLOW RESULTS")
    lines.append("=" * 80)

    for i, item in enumerate(expanded, 1):
        seed = item["seed"]

        lines.append(f"\n{i}. SEED NODE (Hybrid Search Match):")
        lines.append(f"   Node: {seed['node_id']}")
        lines.append(f"   Semantic Score: {seed['similarity']:.3f}")
        lines.append(f"   Text: {seed['text'][:150]}...")

        if seed["data"].get("type") == "control":
            lines.append(f"   Domain: {seed['data'].get('domain', 'N/A')}")
            lines.append(f"   Topic: {seed['data'].get('topic', 'N/A')}")

        lines.append("\n   HYBRID-FILTERED NEIGHBORS:")
        lines.append("   (Must pass BOTH structural AND semantic thresholds)")

        for j, neighbor in enumerate(item["hybrid_neighbors"][:5], 1):
            lines.append(
                f"   {j}. {neighbor['node_id']} ({neighbor['direction']}, {neighbor['hop_distance']} hops)"
            )
            lines.append(f"      Combined: {neighbor['combined_score']:.3f}")
            lines.append(
                f"      └─ Structural: {neighbor['structural_similarity']:.3f}"
            )
            lines.append(f"      └─ Semantic: {neighbor['semantic_similarity']:.3f}")
            lines.append(f"      Type: {neighbor['data'].get('type', 'unknown')}")
            lines.append(f"      {neighbor['text'][:80]}...")

    if subgraph:
        lines.append(f"\n{'=' * 80}")
        lines.append("SUBGRAPH SUMMARY:")
        lines.append(f"  Nodes: {subgraph.number_of_nodes()}")
        lines.append(f"  Edges: {subgraph.number_of_edges()}")

        # Node breakdown
        seed_count = sum(
            1 for n in subgraph.nodes() if subgraph.nodes[n].get("is_seed", False)
        )
        intermediate_count = sum(
            1
            for n in subgraph.nodes()
            if subgraph.nodes[n].get("is_intermediate", False)
        )
        neighbor_count = subgraph.number_of_nodes() - seed_count - intermediate_count

        lines.append(f"  - Seed nodes: {seed_count}")
        lines.append(f"  - Found neighbors: {neighbor_count}")
        lines.append(f"  - Intermediate nodes: {intermediate_count}")

    return "\n".join(lines)
