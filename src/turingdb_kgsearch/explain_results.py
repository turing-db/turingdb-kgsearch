from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def explain_retrieval(
    node_id,
    query,
    subgraph,
    node_vectors,
    sparse_vectors,
    sparse_vectorizer,
    structural_vectors,
    model,
):
    """
    Explain why a node was retrieved.

    Args:
        node_id: Node to explain
        query: Original search query
        subgraph: NetworkX subgraph containing the node
        node_vectors: Dense embeddings dict
        sparse_vectors: Sparse embeddings dict
        sparse_vectorizer: TF-IDF vectorizer
        structural_vectors: Node2Vec embeddings dict
        model: SentenceTransformer model

    Returns:
        Dictionary with explanation components
    """
    import numpy as np

    if node_id not in subgraph:
        return {"error": f"Node {node_id} not in subgraph"}

    node_data = subgraph.nodes[node_id]
    explanation = {
        "node_id": node_id,
        "node_type": node_data.get("type", "unknown"),
        "query": query,
    }

    # 1. Check if it's a seed node
    if node_data.get("is_seed", False):
        explanation["reason"] = "SEED NODE"
        explanation["seed_score"] = node_data.get("seed_score", 0)
        explanation["dense_score"] = node_data.get("dense_score")
        explanation["sparse_score"] = node_data.get("sparse_score")

    # 2. Check if it's a found neighbor
    elif node_data.get("structural_similarity") is not None:
        explanation["reason"] = "FOUND VIA GRAPH TRAVERSAL"
        explanation["seed_node"] = node_data.get("seed_node")
        explanation["hop_distance"] = node_data.get("hop_distance")
        explanation["direction"] = node_data.get("direction")
        explanation["structural_similarity"] = node_data.get("structural_similarity")
        explanation["semantic_similarity"] = node_data.get("semantic_similarity")
        explanation["combined_score"] = node_data.get("combined_score")

    # 3. Check if intermediate
    elif node_data.get("is_intermediate", False):
        explanation["reason"] = "INTERMEDIATE NODE"
        explanation["note"] = "On path between seed and found nodes"

    # 4. Compute detailed breakdown

    # Semantic similarity breakdown
    if node_id in node_vectors:
        query_vector = model.encode([query])[0]
        node_vector = node_vectors[node_id]

        semantic_sim = np.dot(query_vector, node_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(node_vector)
        )

        explanation["semantic"] = {
            "similarity": float(semantic_sim),
            "interpretation": _interpret_similarity(semantic_sim),
        }

    # Keyword matching breakdown
    if node_id in sparse_vectors and sparse_vectorizer:
        query_sparse = sparse_vectorizer.transform([query])
        node_sparse = sparse_vectors[node_id]

        sparse_sim = cosine_similarity(query_sparse, node_sparse)[0][0]

        # Find matching keywords
        query_terms = set(query.lower().split())
        node_text = subgraph.nodes[node_id].get(
            "statement", subgraph.nodes[node_id].get("name", "")
        )
        node_terms = set(node_text.lower().split())

        matching_terms = query_terms & node_terms

        # Get top TF-IDF terms for this node
        feature_names = sparse_vectorizer.get_feature_names_out()
        node_sparse_array = node_sparse.toarray()[0]
        top_indices = node_sparse_array.argsort()[-10:][::-1]
        top_terms = [
            (feature_names[i], node_sparse_array[i])
            for i in top_indices
            if node_sparse_array[i] > 0
        ]

        explanation["keyword"] = {
            "similarity": float(sparse_sim),
            "matching_terms": list(matching_terms),
            "top_tfidf_terms": top_terms[:5],
            "interpretation": _interpret_similarity(sparse_sim),
        }

    # Structural similarity (if available)
    if node_id in structural_vectors:
        # Find which nodes it's structurally similar to
        similar_nodes = []
        for other_id, other_vec in structural_vectors.items():
            if other_id == node_id:
                continue

            # Check if other_id exists in subgraph before accessing
            if other_id not in subgraph:
                continue

            # Only check seed nodes
            if subgraph.nodes[other_id].get("is_seed", False):
                sim = np.dot(structural_vectors[node_id], other_vec) / (
                    np.linalg.norm(structural_vectors[node_id])
                    * np.linalg.norm(other_vec)
                )
                if sim > 0.5:  # Threshold for "similar"
                    similar_nodes.append((other_id, float(sim)))

        similar_nodes.sort(key=lambda x: x[1], reverse=True)

        explanation["structural"] = {
            "similar_to_seeds": similar_nodes[:3],
            "interpretation": "Shares graph structure with seed nodes",
        }

    # 5. Path explanation (if not a seed)
    if not node_data.get("is_seed", False):
        seed_nodes = [
            n for n in subgraph.nodes() if subgraph.nodes[n].get("is_seed", False)
        ]

        paths = []
        for seed in seed_nodes:
            try:
                raise NotImplementedError(
                    "find shortest path not implemented yet in turingdb-graphrag package"
                )
                # path = nx.shortest_path(subgraph, seed, node_id)
                # path_info = _describe_path(path, subgraph)
                # paths.append({
                #    'from_seed': seed,
                #    'length': len(path) - 1,
                #    'path': path,
                #    'path_description': path_info
                # })
            except (nx.NetworkXNoPath, NotImplementedError):
                # print(f"error {e}")
                pass

        if paths:
            # Show shortest path
            shortest = min(paths, key=lambda x: x["length"])
            explanation["path_from_seed"] = shortest

    # 6. Node content preview
    explanation["content_preview"] = _get_content_preview(node_data)

    return explanation


def _interpret_similarity(score):
    """Interpret similarity score in human terms."""
    if score >= 0.9:
        return "Very high similarity"
    elif score >= 0.7:
        return "High similarity"
    elif score >= 0.5:
        return "Moderate similarity"
    elif score >= 0.3:
        return "Low similarity"
    else:
        return "Very low similarity"


def _describe_path(path, subgraph):
    """Create human-readable path description."""
    descriptions = []

    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]

        source_type = subgraph.nodes[source].get("type", "node")
        target_type = subgraph.nodes[target].get("type", "node")

        edge_data = subgraph.get_edge_data(source, target)
        rel = edge_data.get("rel", "connected to") if edge_data else "connected to"

        descriptions.append(
            f"{source} ({source_type}) --[{rel}]--> {target} ({target_type})"
        )

    return descriptions


def _get_content_preview(node_data, max_length=200):
    """Get preview of node content."""
    for key in ["statement", "text", "name", "description"]:
        if key in node_data and node_data[key]:
            text = str(node_data[key])
            if len(text) > max_length:
                return text[:max_length] + "..."
            return text
    return "No text content"


def print_explanation(explanation, verbose=True):
    """Pretty print retrieval explanation."""

    print("\n" + "=" * 80)
    print(f"RETRIEVAL EXPLANATION: {explanation['node_id']}")
    print("=" * 80)

    print(f"\nNode Type: {explanation['node_type']}")
    print(f"Query: '{explanation['query']}'")

    # Main reason
    print(f"\n🎯 Reason: {explanation['reason']}")

    if explanation["reason"] == "SEED NODE":
        print(f"   Initial search score: {explanation['seed_score']:.3f}")
        if explanation.get("dense_score"):
            print(f"   - Semantic component: {explanation['dense_score']:.3f}")
        if explanation.get("sparse_score"):
            print(f"   - Keyword component: {explanation['sparse_score']:.3f}")

    elif explanation["reason"] == "FOUND VIA GRAPH TRAVERSAL":
        print(f"   From seed: {explanation['seed_node']}")
        print(
            f"   Distance: {explanation['hop_distance']} hops ({explanation['direction']})"
        )
        print(f"   Structural similarity: {explanation['structural_similarity']:.3f}")
        print(f"   Semantic similarity: {explanation['semantic_similarity']:.3f}")
        print(f"   Combined score: {explanation['combined_score']:.3f}")

    # Semantic breakdown
    if "semantic" in explanation:
        print(f"\n📝 Semantic Similarity: {explanation['semantic']['similarity']:.3f}")
        print(f"   {explanation['semantic']['interpretation']}")

    # Keyword breakdown
    if "keyword" in explanation and verbose:
        print(f"\n🔍 Keyword Matching: {explanation['keyword']['similarity']:.3f}")
        print(f"   {explanation['keyword']['interpretation']}")

        if explanation["keyword"]["matching_terms"]:
            print(
                f"   Matching terms: {', '.join(explanation['keyword']['matching_terms'])}"
            )

        if explanation["keyword"]["top_tfidf_terms"]:
            print("   Top TF-IDF terms in node:")
            for term, score in explanation["keyword"]["top_tfidf_terms"][:3]:
                print(f"      - {term}: {score:.3f}")

    # Structural
    if "structural" in explanation and verbose:
        print("\n🕸️  Structural Similarity:")
        print(f"   {explanation['structural']['interpretation']}")
        if explanation["structural"]["similar_to_seeds"]:
            print("   Similar to seed nodes:")
            for node, sim in explanation["structural"]["similar_to_seeds"]:
                print(f"      - {node}: {sim:.3f}")

    # Path
    if "path_from_seed" in explanation and verbose:
        path_info = explanation["path_from_seed"]
        print(
            f"\n🛤️  Path from seed '{path_info['from_seed']}' ({path_info['length']} hops):"
        )
        for step in path_info["path_description"]:
            print(f"   {step}")

    # Content preview
    if "content_preview" in explanation:
        print("\n📄 Content Preview:")
        print(f"   {explanation['content_preview']}")

    print("\n" + "=" * 80)


def explain_top_results(
    query,
    results,
    subgraph,
    node_vectors,
    sparse_vectors,
    sparse_vectorizer,
    structural_vectors,
    model,
    top_k=3,
):
    """
    Explain top k search results.

    Args:
        query: Search query
        results: List of search results (from search functions)
        subgraph: NetworkX subgraph
        node_vectors, sparse_vectors, etc.: Embedding dictionaries
        top_k: Number of results to explain

    Returns:
        List of explanations
    """

    explanations = []

    for result in results[:top_k]:
        node_id = result["node_id"]

        explanation = explain_retrieval(
            node_id,
            query,
            subgraph,
            node_vectors,
            sparse_vectors,
            sparse_vectorizer,
            structural_vectors,
            model,
        )

        explanations.append(explanation)
        print_explanation(explanation, verbose=True)

    return explanations
