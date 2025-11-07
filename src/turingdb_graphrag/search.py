import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def dense_search(query, node_vectors, node_texts, G, model, k=5, node_type=None):
    """
    Search for nodes relevant to the query.

    Args:
        query: Natural language question or search term
        node_vectors: Dictionary of {node_id: vector}
        node_texts: Dictionary of {node_id: text}
        G: NetworkX graph
        model: SentenceTransformer model for encoding
        k: Number of results to return
        node_type: Filter by node type ('control', 'topic', 'domain', or None for all)

    Returns:
        List of results with node info and similarity scores
    """
    if not node_vectors:
        return []

    # Encode query to vector
    query_vector = model.encode([query])[0]

    # Calculate similarity to all nodes
    results = []
    for node_id, node_vector in node_vectors.items():
        # Cosine similarity
        similarity = np.dot(query_vector, node_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(node_vector)
        )
        results.append(
            {
                "node_id": node_id,
                "similarity": float(similarity),
                "text": node_texts[node_id],
                "data": dict(G.nodes[node_id]),
            }
        )

    # Filter by node type if specified
    if node_type is not None:
        results = [r for r in results if r["data"].get("type") == node_type]

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:k]


def sparse_search(
    query, sparse_vectors, sparse_vectorizer, node_texts, G, k=5, node_type=None
):
    """
    Search for nodes using sparse (keyword-based) vectors.

    Args:
        query: Natural language question or search term
        sparse_vectors: Dictionary of {node_id: sparse_vector}
        sparse_vectorizer: Fitted TfidfVectorizer for encoding queries
        node_texts: Dictionary of {node_id: text}
        G: NetworkX graph
        k: Number of results to return
        node_type: Filter by node type ('control', 'topic', 'domain', or None for all)

    Returns:
        List of results with node info and similarity scores
    """
    if not sparse_vectors:
        return []

    # Encode query to sparse vector
    query_sparse = sparse_vectorizer.transform([query])

    # Calculate similarity to all nodes
    results = []
    for node_id, node_sparse in sparse_vectors.items():
        # Cosine similarity for sparse vectors
        similarity = cosine_similarity(query_sparse, node_sparse)[0][0]

        results.append(
            {
                "node_id": node_id,
                "similarity": float(similarity),
                "text": node_texts[node_id],
                "data": dict(G.nodes[node_id]),
            }
        )

    # Filter by node type if specified
    if node_type is not None:
        results = [r for r in results if r["data"].get("type") == node_type]

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:k]


def print_results(results):
    """Pretty print search results."""
    print(f"\nFound {len(results)} results:")
    print("=" * 80)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Similarity: {r['similarity']:.4f}")
        print(f"   Node: {r['node_id']}")
        print(f"   Type: {r['data'].get('type', 'unknown')}")
        print(f"   Text: {r['text'][:200]}...")  # First 200 chars
        if r["data"].get("type") == "control":
            print(f"   Domain: {r['data'].get('domain', 'N/A')}")
            print(f"   Topic: {r['data'].get('topic', 'N/A')}")


def hybrid_search(
    query,
    node_vectors,
    node_texts,
    G,
    sparse_vectors,
    sparse_vectorizer,
    model,
    k=5,
    alpha=0.7,
    node_type=None,
):
    """
    Hybrid search combining dense (semantic) and sparse (keyword) search.

    Args:
        query: Search query
        node_vectors: Dictionary of dense vectors
        node_texts: Dictionary of node texts
        G: NetworkX graph
        sparse_vectors: Dictionary of sparse vectors
        sparse_vectorizer: TF-IDF vectorizer
        model: SentenceTransformer model
        k: Number of results
        alpha: Weight (1.0=dense only, 0.0=sparse only, 0.7=balanced)
        node_type: Filter by node type

    Returns:
        List of results with combined scores
    """

    # Get results from both methods (get more to ensure coverage)
    dense_results = dense_search(
        query, node_vectors, node_texts, G, model, k=k * 3, node_type=node_type
    )
    sparse_results = sparse_search(
        query,
        sparse_vectors,
        sparse_vectorizer,
        node_texts,
        G,
        k=k * 3,
        node_type=node_type,
    )

    # Build score dictionaries
    dense_scores = {r["node_id"]: r["similarity"] for r in dense_results}
    sparse_scores = {r["node_id"]: r["similarity"] for r in sparse_results}

    # Normalize scores to [0, 1]
    def normalize(scores):
        if not scores:
            return {}
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        if max_val - min_val < 1e-8:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    dense_norm = normalize(dense_scores)
    sparse_norm = normalize(sparse_scores)

    # Combine scores
    all_nodes = set(dense_norm.keys()) | set(sparse_norm.keys())

    combined = []
    for node_id in all_nodes:
        d_score = dense_norm.get(node_id, 0.0)
        s_score = sparse_norm.get(node_id, 0.0)
        final_score = alpha * d_score + (1 - alpha) * s_score

        combined.append(
            {
                "node_id": node_id,
                "similarity": final_score,
                "dense_score": d_score,
                "sparse_score": s_score,
                "text": node_texts[node_id],
                "data": dict(G.nodes[node_id]),
            }
        )

    # Sort and return top k
    combined.sort(key=lambda x: x["similarity"], reverse=True)
    return combined[:k]


def print_hybrid_results(results):
    """Pretty print hybrid search results showing both scores."""
    print(f"\nFound {len(results)} results:")
    print("=" * 80)
    for i, r in enumerate(results, 1):
        print(
            f"\n{i}. Combined: {r['similarity']:.3f} (Dense: {r['dense_score']:.3f}, Sparse: {r['sparse_score']:.3f})"
        )
        print(f"   Node: {r['node_id']}")
        print(f"   Type: {r['data'].get('type', 'unknown')}")
        print(f"   Text: {r['text'][:150]}...")


def compare_search_methods(
    query,
    node_vectors,
    node_texts,
    G,
    sparse_vectors,
    sparse_vectorizer,
    model,
    k=5,
    alpha=0.7,
    node_type="control",
):
    """
    Compare dense, sparse, and hybrid search side-by-side.

    Args:
        query: Search query
        node_vectors: Dense vectors
        node_texts: Node texts
        sparse_vectors: Sparse vectors
        sparse_vectorizer: TF-IDF vectorizer
        model: SentenceTransformer model
        k: Number of results
        alpha: Alpha for hybrid (default 0.7)
        node_type: Filter by node type
    """

    print(f"\n{'='*80}")
    print(f"QUERY: '{query}'")
    print("=" * 80)

    # 1. Dense only
    print("\n1. DENSE ONLY (Semantic):")
    print("-" * 80)
    results_dense = dense_search(
        query, node_vectors, node_texts, G, model, k, node_type
    )
    for i, r in enumerate(results_dense[:3], 1):
        print(f"{i}. {r['similarity']:.3f} | {r['text'][:100]}...")

    # 2. Sparse only
    print("\n2. SPARSE ONLY (Keywords):")
    print("-" * 80)
    results_sparse = sparse_search(
        query, sparse_vectors, sparse_vectorizer, node_texts, G, k, node_type
    )
    for i, r in enumerate(results_sparse[:3], 1):
        print(f"{i}. {r['similarity']:.3f} | {r['text'][:100]}...")

    # 3. Hybrid
    print(f"\n3. HYBRID (alpha={alpha}):")
    print("-" * 80)
    results_hybrid = hybrid_search(
        query,
        node_vectors,
        node_texts,
        G,
        sparse_vectors,
        sparse_vectorizer,
        model,
        k,
        alpha,
        node_type,
    )
    for i, r in enumerate(results_hybrid[:3], 1):
        print(
            f"{i}. {r['similarity']:.3f} (D:{r['dense_score']:.2f}/S:{r['sparse_score']:.2f}) | {r['text'][:100]}..."
        )
