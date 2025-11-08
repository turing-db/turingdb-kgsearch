# TuringDB KGSearch

Knowledge Graph Search: Semantic search and graph operations combining vector embeddings with graph structure.

## Overview

TuringDB KGSearch combines **knowledge graph structures** with **vector embeddings** to enable semantic search that understands both content and relationships.

**How it differs from traditional vector search:**

Traditional vector databases embed each document independently - a document about "data encryption" is just a point in vector space, disconnected from related concepts. This loses critical information:
- **Hierarchical structure**: Which domain or category does it belong to?
- **Cross-references**: What other documents cite or depend on it?
- **Metadata relationships**: Which standards, regulations, or frameworks does it map to?

KGSearch preserves this context by embedding the graph structure itself. When you search for "encryption", KGSearch knows:
- This control belongs to the "Data Security" domain
- It relates to the "Privacy" topic
- It maps to ISO27001 A.8.24, NIST RMF SC-28, and SOC2 CC6.1
- It has prerequisites and dependencies

**Example:** A control about "encryption" in traditional vector search returns isolated text matches. In KGSearch, you get the control plus its domain ("Information Security"), topic ("Data Security"), and all mapped standards (ISO27001 A.8.24, NIST RMF SC-28), enabling queries like "find all ISO27001 encryption controls" even when "ISO27001" doesn't appear in the control text.

### Two-Stage Retrieval

KGSearch uses a sophisticated two-stage approach:

**Stage 1: Seed Discovery (Hybrid Search)**
- Combines semantic search (dense embeddings) with keyword matching (sparse embeddings)
- Finds initial relevant nodes using configurable α parameter (semantic vs. keyword weight)

**Stage 2: Graph Traversal with Topological Filtering**
- Explores neighborhood around seed nodes
- Uses **topological similarity** (Node2Vec structural embeddings) combined with semantic relevance to decide which graph branches to explore
- Prunes irrelevant paths dynamically, keeping only contextually related nodes
- Results in a focused subgraph containing both direct matches and structurally related content

This enables queries that leverage both meaning and structure: "Find all controls related to AI risk management and their mapped standards" returns not just matching controls, but the full regulatory context.

## Features

- **Hybrid Search**: Combine semantic (dense) and keyword (sparse) search with configurable weighting
- **Context-Aware Embeddings**: Incorporate graph structure into vector representations
- **Two-Stage Retrieval**: Seed discovery + topological graph traversal
- **Node Importance Ranking**: PageRank, betweenness, and custom graph metrics
- **Similarity Analysis**: Compare nodes using vector similarity
- **Interactive Visualization**: PyVis-based graph exploration
- **LLM Integration**: Generate prompts with graph context and query LLMs for RAG applications
- **Result Explanation**: Understand why nodes were retrieved
- **Scalable**: Designed for large graphs with millions of nodes and edges

## Installation

```bash
# Using uv (recommended)
uv add turingdb-kgsearch

# Using pip
pip install turingdb-kgsearch
```

## Quick Start

```python
import networkx as nx
from turingdb_kgsearch.embeddings import build_smart_enriched_embeddings
from turingdb_kgsearch.search import hybrid_search, print_results
from turingdb_kgsearch.workflow import search_and_expand_hybrid_filtered
from sentence_transformers import SentenceTransformer

# Create a knowledge graph
G = nx.DiGraph()
G.add_node("doc1", type="document", content="Machine learning basics")
G.add_node("doc2", type="document", content="Deep learning fundamentals")
G.add_node("topic1", type="topic", name="AI Fundamentals")
G.add_edge("topic1", "doc1", rel="contains")
G.add_edge("topic1", "doc2", rel="contains")
G.add_edge("doc1", "doc2", rel="prerequisite")

# Load embedding model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Build embeddings with graph context
node_vectors, node_texts = build_smart_enriched_embeddings(G, model)

# Simple search
results = hybrid_search("introduction to ML", node_vectors, node_texts, k=5, alpha=0.7)
print_results(results)

# Advanced: Two-stage retrieval with graph traversal
subgraph, results = search_and_expand_hybrid_filtered(
    query="neural networks",
    G=G,
    node_vectors=node_vectors,
    node_texts=node_texts,
    k_seeds=5,           # Top 5 seed nodes
    max_depth=2,         # Explore 2 hops
    alpha=0.7,           # 70% semantic, 30% keyword
    node_type='document' # Filter by node type
)
```

## Key Concepts

### Embedding Strategies

```python
from turingdb_kgsearch.embeddings import (
    build_node_only_embeddings,        # Node attributes only
    build_context_enriched_embeddings, # Include 1-hop neighbors
    build_smart_enriched_embeddings,   # Type-specific context (recommended)
    build_sparse_embeddings,           # TF-IDF for keyword search
    build_node2vec_embeddings,         # Structural/topological embeddings
)
```

**Node-Only**
- Embeds only node attributes
- Fast, minimal memory
- Best for simple content matching

**Context-Enriched**
- Includes 1-hop neighborhood information
- Captures hierarchical relationships
- Better semantic understanding

**Smart** (recommended)
- Type-specific context enrichment
- Balances content and structure
- Requires domain knowledge to select which node properties to include in embeddings
- Optimal for most use cases

**Node2Vec**
- Captures graph topology and node proximity
- Used for graph traversal filtering
- Complements semantic embeddings

### Hybrid Search

Balance semantic understanding with keyword precision:

```python
from turingdb_kgsearch.search import hybrid_search

# Pure semantic (α=1.0): conceptual queries
results = hybrid_search(query, node_vectors, node_texts, alpha=1.0, k=5)

# Balanced (α=0.7): general use (recommended)
results = hybrid_search(query, node_vectors, node_texts, alpha=0.7, k=5)

# Pure keyword (α=0.0): exact term matching
results = hybrid_search(query, node_vectors, node_texts, alpha=0.0, k=5)
```

### Two-Stage Retrieval Workflow

Combine search with graph traversal:

```python
from turingdb_kgsearch.workflow import search_and_expand_hybrid_filtered

# Stage 1: Find seed nodes via hybrid search
# Stage 2: Expand using topological + semantic filtering
subgraph, results = search_and_expand_hybrid_filtered(
    query="risk management",
    G=G,
    node_vectors=node_vectors,
    node_texts=node_texts,
    structural_vectors=node2vec_vectors,  # Optional: topological filtering
    k_seeds=5,
    max_depth=2,
    alpha=0.7,
    node_type='control'
)
```

## Architecture

### Why Knowledge Graph + Vectors?

Traditional vector search treats documents as isolated points. KGSearch preserves relationships:

- **Hierarchical context**: Parent-child relationships inform embeddings
- **Cross-references**: Find related content through graph edges
- **Multi-hop reasoning**: Traverse relationships to discover indirect connections
- **Structured metadata**: Leverage graph attributes alongside content

### When to Use KGSearch

✅ **Good fit:**
- Hierarchical content (documents → sections → paragraphs)
- Cross-referenced material (standards, regulations, citations)
- Knowledge bases with explicit relationships
- Multi-aspect queries requiring context
- Large-scale graphs (millions of nodes/edges)

❌ **Overkill for:**
- Flat document collections without relationships
- Simple keyword search requirements

## Example: AI Governance Control Mappings

See `notebooks/ai_governance_example.ipynb` for a complete example mapping regulatory controls across multiple standards (ISO, NIST, EU AI Act, SOC2).

**Key features demonstrated:**
- Hierarchical graph: Domain → Topic → Control → Standards
- Semantic search for compliance requirements
- Cross-standard overlap analysis
- Interactive visualization

**Example query:**
```python
from turingdb_kgsearch.search import hybrid_search
from turingdb_kgsearch.workflow import search_and_expand_hybrid_filtered

# Find controls related to data privacy across all standards
results = hybrid_search("data privacy protection", node_vectors, node_texts, k=5)

# Two-stage retrieval: find controls + expand to related standards
subgraph, results = search_and_expand_hybrid_filtered(
    query="data privacy protection",
    G=G,
    node_vectors=node_vectors,
    node_texts=node_texts,
    k_seeds=3,
    max_depth=2,
    node_type='control'
)
```

## Development

```bash
# Clone the repository
git clone https://github.com/turing-db/turingdb-kgsearch.git
cd turingdb-kgsearch

# Install with development dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check
```

## Requirements

- Python ≥3.13
- NetworkX ≥3.5
- sentence-transformers ≥5.1
- scikit-learn ≥1.7
- numpy ≥2.3
- pandas ≥2.3

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Citation

If you use TuringDB KGSearch in research, please cite:

```bibtex
@software{turingdb_kgsearch,
  title = {TuringDB KGSearch: Knowledge Graph Search with Vector Embeddings},
  author = {TuringDB Team},
  year = {2025},
  url = {https://github.com/turing-db/turingdb-kgsearch}
}
```
