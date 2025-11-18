def build_node_only_embeddings(G, model):
    """Current approach: Only encode node's own attributes."""

    node_vectors = {}
    node_texts = {}
    texts_to_encode = []
    node_ids = []

    for node_id, data in G.nodes(data=True):
        node_type = data.get("type", "")

        if node_type == "control":
            text = data.get("statement", "")
        elif node_type == "topic":
            text = f"Topic: {data.get('name', '')}"
        elif node_type == "domain":
            text = f"Domain: {data.get('name', '')}"
        elif node_type == "standard":
            text = f"{data.get('standard', '')} {data.get('reference', '')}"
        else:
            text = str(node_id)

        if text and text.strip():
            texts_to_encode.append(text)
            node_ids.append(node_id)
            node_texts[node_id] = text

    embeddings = model.encode(texts_to_encode, show_progress_bar=False)

    for node_id, vector in zip(node_ids, embeddings):
        node_vectors[node_id] = vector

    print(
        f"✓ Vector index built using node-only embeddings approach: {len(node_vectors)} vectors"
    )

    return node_vectors, node_texts


def build_context_enriched_embeddings(G, model, strategy="lightweight"):
    """
    Build embeddings that incorporate neighborhood context.

    Strategies:
    - 'lightweight': Add parent/child context (1-hop)
    - 'moderate': Add full 1-hop neighborhood with edge types
    - 'heavy': Add 2-hop neighborhood (can get very long)
    """

    node_vectors = {}
    node_texts = {}
    texts_to_encode = []
    node_ids = []

    for node_id, data in G.nodes(data=True):
        node_type = data.get("type", "")

        # Start with node's own content
        if node_type == "control":
            base_text = data.get("statement", "")
        elif node_type == "topic":
            base_text = f"Topic: {data.get('name', '')}"
        elif node_type == "domain":
            base_text = f"Domain: {data.get('name', '')}"
        elif node_type == "standard":
            base_text = f"{data.get('standard', '')} {data.get('reference', '')}"
        else:
            base_text = str(node_id)

        # Add contextual information based on strategy
        context_parts = [base_text]

        if strategy in ["lightweight", "moderate", "heavy"]:
            # Add parent context (predecessors)
            for pred in G.predecessors(node_id):
                pred_data = G.nodes[pred]
                pred_type = pred_data.get("type", "")

                if pred_type == "domain":
                    context_parts.append(f"in domain {pred_data.get('name', '')}")
                elif pred_type == "topic":
                    context_parts.append(f"under topic {pred_data.get('name', '')}")

        if strategy in ["moderate", "heavy"]:
            # Add child context (successors)
            for succ in G.successors(node_id):
                succ_data = G.nodes[succ]
                succ_type = succ_data.get("type", "")

                if succ_type == "standard":
                    std_name = succ_data.get("standard", "")
                    std_ref = succ_data.get("reference", "")
                    context_parts.append(f"maps to {std_name} {std_ref}")
                elif succ_type == "control":
                    # For topics/domains, include control statements
                    statement = succ_data.get("statement", "")[:100]  # Truncate
                    if statement:
                        context_parts.append(f"includes: {statement}")

        if strategy == "heavy":
            # Add 2-hop neighborhood (can get very long!)
            for pred in G.predecessors(node_id):
                for pred2 in G.predecessors(pred):
                    pred2_data = G.nodes[pred2]
                    if pred2_data.get("type") == "domain":
                        context_parts.append(f"related to {pred2_data.get('name', '')}")

        # Combine all context
        enriched_text = ". ".join(context_parts)

        if enriched_text and enriched_text.strip():
            texts_to_encode.append(enriched_text)
            node_ids.append(node_id)
            node_texts[node_id] = enriched_text

    embeddings = model.encode(texts_to_encode, show_progress_bar=False)

    for node_id, vector in zip(node_ids, embeddings):
        node_vectors[node_id] = vector

    print(
        f"✓ Vector index built using context-enriched embeddings approach (strategy {strategy}): {len(node_vectors)} vectors"
    )

    return node_vectors, node_texts


def build_smart_enriched_embeddings(G, model):
    """
    Intelligent context enrichment based on node type.
    Different nodes benefit from different context.
    """

    node_vectors = {}
    node_texts = {}
    texts_to_encode = []
    node_ids = []

    for node_id, data in G.nodes(data=True):
        node_type = data.get("type", "")

        # Build context based on what's actually useful for each type
        if node_type == "control":
            # For controls: Include parent topic/domain AND mapped standards
            base_text = data.get("statement", "")
            context_parts = [base_text]

            # Add hierarchical context
            for pred in G.predecessors(node_id):
                pred_data = G.nodes[pred]
                pred_type = pred_data.get("type", "")
                if pred_type == "topic":
                    context_parts.append(f"Topic: {pred_data.get('name', '')}")
                    # Also get domain from topic's parent
                    for pred2 in G.predecessors(pred):
                        pred2_data = G.nodes[pred2]
                        if pred2_data.get("type") == "domain":
                            context_parts.append(
                                f"Domain: {pred2_data.get('name', '')}"
                            )

            # Add standard mappings (VERY useful for search!)
            standards = []
            for succ in G.successors(node_id):
                succ_data = G.nodes[succ]
                if succ_data.get("type") == "standard":
                    std_name = succ_data.get("standard", "")
                    std_ref = succ_data.get("reference", "")
                    standards.append(f"{std_name} {std_ref}")

            if standards:
                context_parts.append(f"Standards: {', '.join(standards)}")

            enriched_text = ". ".join(context_parts)

        elif node_type == "topic":
            # For topics: Include domain AND sample control statements
            base_text = f"Topic: {data.get('name', '')}"
            context_parts = [base_text]

            # Add domain
            for pred in G.predecessors(node_id):
                pred_data = G.nodes[pred]
                if pred_data.get("type") == "domain":
                    context_parts.append(f"Domain: {pred_data.get('name', '')}")

            # Add sample controls (helps understand what this topic covers)
            control_samples = []
            for succ in G.successors(node_id):
                succ_data = G.nodes[succ]
                if succ_data.get("type") == "control":
                    statement = succ_data.get("statement", "")[:80]  # Short sample
                    if statement and len(control_samples) < 3:  # Limit to 3 samples
                        control_samples.append(statement)

            if control_samples:
                context_parts.append(f"Examples: {'; '.join(control_samples)}")

            enriched_text = ". ".join(context_parts)

        elif node_type == "domain":
            # For domains: Include topics and high-level summary
            base_text = f"Domain: {data.get('name', '')}"
            context_parts = [base_text]

            # Add topics
            topics = []
            for succ in G.successors(node_id):
                succ_data = G.nodes[succ]
                if succ_data.get("type") == "topic":
                    topics.append(succ_data.get("name", ""))

            if topics:
                context_parts.append(f"Topics: {', '.join(topics)}")

            enriched_text = ". ".join(context_parts)

        elif node_type == "standard":
            # For standards: Include what controls it maps to
            std_name = data.get("standard", "")
            std_ref = data.get("reference", "")
            base_text = f"{std_name} {std_ref}"
            context_parts = [base_text]

            # Add ONE control sample (standards are often queried by reference)
            for pred in G.predecessors(node_id):
                pred_data = G.nodes[pred]
                if pred_data.get("type") == "control":
                    statement = pred_data.get("statement", "")[:100]
                    if statement:
                        context_parts.append(f"Control: {statement}")
                        break  # Just one sample is enough

            enriched_text = ". ".join(context_parts)

        else:
            enriched_text = str(node_id)

        if enriched_text and enriched_text.strip():
            texts_to_encode.append(enriched_text)
            node_ids.append(node_id)
            node_texts[node_id] = enriched_text

    embeddings = model.encode(texts_to_encode, show_progress_bar=False)

    for node_id, vector in zip(node_ids, embeddings):
        node_vectors[node_id] = vector

    print(
        f"✓ Vector index built using type-specific context enrichment approach: {len(node_vectors)} vectors"
    )

    return node_vectors, node_texts


def build_sparse_embeddings(G, node_texts=None, max_features=500, ngram_range=(1, 2)):
    """
    Build sparse (keyword-based) embeddings using TF-IDF.

    Args:
        G: NetworkX graph
        node_texts: Dictionary of {node_id: text} from dense embeddings (optional)
                   If None, extracts texts from graph attributes
        max_features: Maximum vocabulary size
        ngram_range: Tuple of (min_n, max_n) for n-grams

    Returns:
        Tuple of (sparse_vectors, node_texts, sparse_vectorizer)
        - sparse_vectors: Dictionary of {node_id: sparse_vector}
        - node_texts: Dictionary of {node_id: text} (same as input if provided)
        - sparse_vectorizer: Fitted TfidfVectorizer (needed for queries)

    Example:
        # Use same texts as dense embeddings (recommended for hybrid search)
        dense_vectors, node_texts = build_smart_enriched_embeddings(G, model)
        sparse_vectors, _, vectorizer = build_sparse_embeddings(G, node_texts=node_texts)

        # Or build from scratch (uses node attributes only)
        sparse_vectors, node_texts, vectorizer = build_sparse_embeddings(G)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("Building sparse index (TF-IDF)...")

    # If node_texts not provided, extract from graph
    if node_texts is None:
        print("  No node_texts provided - extracting from graph attributes")
        node_texts = {}
        for node_id, data in G.nodes(data=True):
            node_type = data.get("type", "")

            if node_type == "control":
                text = data.get("statement", "")
            elif node_type == "topic":
                text = f"Topic: {data.get('name', '')}"
            elif node_type == "domain":
                text = f"Domain: {data.get('name', '')}"
            elif node_type == "standard":
                text = f"{data.get('standard', '')} {data.get('reference', '')}"
            else:
                text = str(node_id)

            if text and text.strip():
                node_texts[node_id] = text
    else:
        print("  Using provided node_texts (from dense embeddings)")

    # Prepare texts for vectorization
    node_ids = list(node_texts.keys())
    texts_to_encode = [node_texts[nid] for nid in node_ids]

    # Create TF-IDF vectorizer
    sparse_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words="english",
        min_df=1,
        token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
    )

    # Build sparse vectors
    sparse_matrix = sparse_vectorizer.fit_transform(texts_to_encode)

    # Store as dictionary
    sparse_vectors = {}
    for i, node_id in enumerate(node_ids):
        sparse_vectors[node_id] = sparse_matrix[i]

    print(f"✓ Sparse index built: {len(sparse_vectors)} vectors")
    print(f"  Vocabulary size: {len(sparse_vectorizer.vocabulary_)}")

    # Show sample terms
    feature_names = sparse_vectorizer.get_feature_names_out()
    print(f"  Sample terms: {list(feature_names[:20])}")

    return sparse_vectors, node_texts, sparse_vectorizer


def build_node2vec_embeddings(G, dimensions=128, walk_length=10, num_walks=80):
    """
    Build structural embeddings using Node2Vec.

    Args:
        G: NetworkX graph
        dimensions: Embedding dimension (match text embeddings)
        walk_length: Length of random walks
        num_walks: Number of walks per node

    Returns:
        Dictionary of {node_id: structural_vector}
    """
    from node2vec import Node2Vec

    print("Training Node2Vec on graph structure...")

    # Create Node2Vec model
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=1,
        q=1,
        workers=2,
        quiet=True,
    )

    # Train
    n2v_model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # Extract vectors
    structural_vectors = {}
    for node in G.nodes():
        try:
            structural_vectors[str(node)] = n2v_model.wv[str(node)]
        except KeyError:
            pass

    print(f"✓ Node2Vec trained: {len(structural_vectors)} structural vectors")

    return structural_vectors
