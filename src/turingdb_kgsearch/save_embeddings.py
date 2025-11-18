"""
Save and load embeddings using NumPy's compressed format.
Efficient for large graphs with millions of nodes.
Supports dense, sparse, and structural embeddings.
"""

import numpy as np
from pathlib import Path
from scipy.sparse import (
    issparse,
    vstack,
    save_npz as scipy_save_npz,
    load_npz as scipy_load_npz,
)
import pickle


# ============================================================================
# SAVE AND LOAD EMBEDDINGS (NUMPY FORMAT)
# ============================================================================


def save_embeddings(
    node_vectors,
    node_texts=None,
    filepath="embeddings.npz",
    embedding_type="dense",
    vectorizer=None,
):
    """
    Save embeddings in NumPy's compressed format.
    Handles both dense and sparse vectors.

    Args:
        node_vectors: Dictionary of {node_id: vector}
                     Vectors can be numpy arrays (dense) or scipy sparse matrices
        node_texts: Dictionary of {node_id: text} (optional, can be None for structural embeddings)
        filepath: Path to save file (should end with .npz)
        embedding_type: Type of embedding ('dense', 'sparse', 'node2vec', 'structural')
        vectorizer: Optional vectorizer object (e.g., TfidfVectorizer for sparse embeddings)

    Example:
        # Dense/sparse embeddings with text
        save_embeddings(node_vectors, node_texts, "embeddings/smart_embeddings.npz", "dense")

        # Node2Vec embeddings without text
        save_embeddings(structural_vectors, None, "embeddings/node2vec.npz", "node2vec")

        # Sparse TF-IDF embeddings with vectorizer
        save_embeddings(sparse_vectors, node_texts, "embeddings/sparse.npz", "sparse",
                       vectorizer=sparse_vectorizer)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Get node IDs
    node_ids = list(node_vectors.keys())

    # Check if vectors are sparse
    first_vector = node_vectors[node_ids[0]]
    is_sparse = issparse(first_vector)

    if is_sparse:
        # Handle sparse matrices
        vectors = vstack([node_vectors[nid] for nid in node_ids])
        vector_dim = vectors.shape[1]

        # Save sparse matrix separately
        sparse_filepath = filepath.replace(".npz", "_sparse.npz")
        scipy_save_npz(sparse_filepath, vectors)

        # Save metadata
        save_data = {
            "node_ids": node_ids,
            "embedding_type": embedding_type,
            "is_sparse": True,
            "vector_dim": vector_dim,
            "sparse_filepath": sparse_filepath,
        }
    else:
        # Handle dense vectors
        vectors = np.array([node_vectors[nid] for nid in node_ids])
        vector_dim = vectors.shape[1]

        save_data = {
            "node_ids": node_ids,
            "vectors": vectors,
            "embedding_type": embedding_type,
            "is_sparse": False,
            "vector_dim": vector_dim,
        }

    # Add texts only if provided
    if node_texts is not None:
        texts = np.array([node_texts[nid] for nid in node_ids])
        save_data["texts"] = texts

    # Save vectorizer if provided (for sparse embeddings)
    if vectorizer is not None:
        vectorizer_filepath = filepath.replace(".npz", "_vectorizer.pkl")
        with open(vectorizer_filepath, "wb") as f:
            pickle.dump(vectorizer, f)
        save_data["vectorizer_filepath"] = vectorizer_filepath
        print(f"  - Vectorizer saved to: {vectorizer_filepath}")

    # Save with compression
    np.savez_compressed(filepath, **save_data)

    print(f"✓ Embeddings saved to: {filepath}")
    print(f"  - Type: {embedding_type}")
    print(f"  - Format: {'Sparse' if is_sparse else 'Dense'}")
    print(f"  - {len(node_vectors)} vectors")
    print(f"  - Vector dimension: {vector_dim}")
    print(f"  - Has texts: {'Yes' if node_texts is not None else 'No'}")
    print(f"  - Has vectorizer: {'Yes' if vectorizer is not None else 'No'}")
    print(f"  - File size: {Path(filepath).stat().st_size / (1024*1024):.2f} MB")
    if is_sparse:
        print(f"  - Sparse file: {sparse_filepath}")


def load_embeddings(filepath="embeddings.npz"):
    """
    Load embeddings from NumPy format.
    Handles all embedding types (dense, sparse, node2vec, structural).

    Args:
        filepath: Path to saved file

    Returns:
        (node_vectors, node_texts, metadata, vectorizer) tuple
        - node_vectors: Dictionary of {node_id: vector}
        - node_texts: Dictionary of {node_id: text} or None if not available
        - metadata: Dictionary with 'embedding_type', 'is_sparse', and other info
        - vectorizer: Vectorizer object (for sparse embeddings) or None

    Example:
        # Dense embeddings (with texts)
        node_vectors, node_texts, meta, _ = load_embeddings("embeddings/smart_embeddings.npz")

        # Node2Vec embeddings (no texts)
        structural_vectors, _, meta, _ = load_embeddings("embeddings/node2vec.npz")

        # Sparse embeddings (with vectorizer)
        sparse_vectors, node_texts, meta, sparse_vectorizer = load_embeddings("embeddings/sparse.npz")
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")

    # Load compressed data
    data = np.load(filepath, allow_pickle=True)

    # Extract common data
    node_ids = data["node_ids"]
    embedding_type = str(data.get("embedding_type", "unknown"))
    is_sparse = bool(data.get("is_sparse", False))
    vector_dim = int(data.get("vector_dim", 0))

    # Load vectors (sparse or dense)
    if is_sparse:
        sparse_filepath = str(data["sparse_filepath"])
        if not Path(sparse_filepath).exists():
            raise FileNotFoundError(f"Sparse matrix file not found: {sparse_filepath}")
        vectors = scipy_load_npz(sparse_filepath)
        # Convert to dictionary of sparse row vectors
        node_vectors = {nid: vectors[i] for i, nid in enumerate(node_ids)}
    else:
        vectors = data["vectors"]
        vector_dim = vectors.shape[1]
        # Convert to dictionary of dense vectors
        node_vectors = {nid: vec for nid, vec in zip(node_ids, vectors)}

    # Load texts if available
    node_texts = None
    if "texts" in data:
        texts = data["texts"]
        node_texts = {nid: str(text) for nid, text in zip(node_ids, texts)}

    # Load vectorizer if available
    vectorizer = None
    if "vectorizer_filepath" in data:
        vectorizer_filepath = str(data["vectorizer_filepath"])
        if Path(vectorizer_filepath).exists():
            with open(vectorizer_filepath, "rb") as f:
                vectorizer = pickle.load(f)
        else:
            print(f"  ⚠️  Vectorizer file not found: {vectorizer_filepath}")

    # Create metadata
    metadata = {
        "embedding_type": embedding_type,
        "is_sparse": is_sparse,
        "num_vectors": len(node_vectors),
        "vector_dim": vector_dim,
        "has_texts": node_texts is not None,
        "has_vectorizer": vectorizer is not None,
    }

    print(f"✓ Embeddings loaded from: {filepath}")
    print(f"  - Type: {embedding_type}")
    print(f"  - Format: {'Sparse' if is_sparse else 'Dense'}")
    print(f"  - {len(node_vectors)} vectors")
    print(f"  - Vector dimension: {vector_dim}")
    print(f"  - Has texts: {'Yes' if node_texts is not None else 'No'}")
    print(f"  - Has vectorizer: {'Yes' if vectorizer is not None else 'No'}")

    return node_vectors, node_texts, metadata, vectorizer
