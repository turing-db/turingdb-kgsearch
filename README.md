# TuringDB KGSearch

Semantic search and graph exploration for knowledge graphs using vector embeddings.

## Overview

TuringDB KGSearch enables semantic search on knowledge graphs by combining:
- **Vector embeddings** for semantic understanding
- **Graph structure** for relationship-aware retrieval
- **Hybrid search** (semantic + keyword matching)

**Traditional vector search** treats each node independently. **KGSearch** preserves context:
- Hierarchical relationships (Domain → Topic → Document)
- Cross-references and dependencies
- Metadata mappings (standards, regulations, citations)

**Example**: Searching "encryption" in a regulatory knowledge graph returns the control text, its domain ("Information Security"), topic ("Data Security"), and all mapped standards (ISO27001, NIST, SOC2) - even when these standards aren't mentioned in the text.

## How It Works

KGSearch combines three complementary techniques:

1. **Dense embeddings** (semantic): Captures meaning using transformer models
2. **Sparse embeddings** (keywords): TF-IDF for exact term matching  
3. **Structural embeddings** (graph): Node2Vec learns from graph topology

**Two-stage retrieval**:
- Stage 1: Hybrid search finds semantically + keyword relevant nodes
- Stage 2: Graph traversal filtered by structural AND semantic similarity

## Installation
```bash
pip install turingdb-kgsearch
```

## Quick Start
```python
import networkx as nx
from sentence_transformers import SentenceTransformer
from turingdb_kgsearch.embeddings import build_context_enriched_embeddings, build_sparse_embeddings, build_node2vec_embeddings
from turingdb_kgsearch.search import hybrid_search

# Create knowledge graph
G = nx.DiGraph()
G.add_node("doc1", type="document", text="Machine learning basics")
G.add_node("doc2", type="document", text="Deep learning neural networks")
G.add_node("topic1", type="topic", name="AI")
G.add_edge("topic1", "doc1", rel="contains")
G.add_edge("topic1", "doc2", rel="contains")
print(G)

# Build embeddings
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
node_vectors_context_heavy, node_texts = build_context_enriched_embeddings(G, model, strategy="heavy")
node_vectors_sparse, _, vectorizer_sparse = build_sparse_embeddings(G, node_texts=node_texts)
node_vectors_node2vec = build_node2vec_embeddings(G, dimensions=384)

# Search
results = hybrid_search(
    query="introduction to neural networks",
    G=G,
    dense_node_vectors=node_vectors_context_heavy,
    sparse_node_vectors=node_vectors_sparse,
    sparse_vectorizer=vectorizer_sparse,
    node_texts=node_texts,  # same node texts used for both dense and sparse
    model=model,
    k=3,
    alpha=0.7,  # 70% semantic, 30% keywords
)

for r in results[:3]:
    print(f"{r['node_id']}: {r['similarity']:.3f}")
```

## Core Features

### 1. Embedding Strategies
```python
from turingdb_kgsearch.embeddings import (
    build_node_only_embeddings,        # Basic: node attributes only
    build_context_enriched_embeddings, # Include neighbor text ("lightweight", "moderate" or "heavy" strategies)
    build_sparse_embeddings,           # Keyword matching (TF-IDF)
    build_node2vec_embeddings          # Structural embeddings
)

# Context-enriched embeddings (best for most use cases)
node_vectors_context_enriched, node_texts = build_context_enriched_embeddings(G, model)

# Build sparse embeddings
node_vectors_sparse, _, vectorizer_sparse = build_sparse_embeddings(
    G=G, node_texts=node_texts
)
```

### 2. Hybrid Search

Balance semantic understanding with keyword precision:
```python
from turingdb_kgsearch.search import hybrid_search

results = hybrid_search(
    query="introduction to neural networks",
    G=G,
    dense_node_vectors=node_vectors_context_enriched,
    sparse_node_vectors=node_vectors_sparse,
    sparse_vectorizer=vectorizer_sparse,
    node_texts=node_texts,  # same node texts used for both dense and sparse
    model=model,
    k=3,  # Number of results
    alpha=0.7,  # 70% semantic, 30% keywords (1.0=semantic only, 0.0=keywords only)
    node_type='control'  # Optional: filter by type
)
```

### 3. Two-Stage Retrieval

Find relevant nodes, then explore their graph neighborhood:
```python
from turingdb_kgsearch.workflow import search_and_expand_hybrid_filtered

# Stage 1: Find seed nodes via hybrid search
# Stage 2: Traverse graph with semantic+structural filtering
semantic_results, expanded, subgraph = search_and_expand_hybrid_filtered(
    query="risk management AI systems",
    G=G,
    node_vectors=node_vectors_context_heavy,
    node_texts=node_texts,
    sparse_vectors=node_vectors_sparse,
    sparse_vectorizer=vectorizer_sparse,
    structural_vectors=node_vectors_node2vec,
    model=model,
    k_search=3,           # Find 3 seed nodes
    max_hops=2,           # Explore 2 hops in graph
    min_structural_sim=0.7,  # Structural similarity threshold
    min_semantic_sim=0.6,    # Semantic similarity threshold
    structural_weight=0.5,  # 50-50 balance (structure vs. semantic)
    alpha=0.7,  # Weight alpha to attribute to semantic (dense) search, (1 - alpha) for keyword (sparse) search
)

# Subgraph contains relevant nodes + edges + similarity scores
print(f"Found {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
```

### 4. Graph Analysis
```python
from turingdb_kgsearch.statistics import get_subgraph_stats, print_subgraph_stats

# Comprehensive statistics
stats = get_subgraph_stats(subgraph, include_node_breakdown=True, include_centrality=True, include_paths=True)
print_subgraph_stats(stats, verbose=True)

from turingdb_kgsearch.ranking import rank_nodes_by_importance, print_node_rankings

# Usage examples
rankings = rank_nodes_by_importance(
    subgraph,
    methods="all",  # or ['pagerank', 'degree', 'relevance']
    top_k=10,
    aggregate="average",  # or 'max' or {'pagerank': 0.4, 'degree': 0.3, 'relevance': 0.3}
)

print_node_rankings(rankings, subgraph, show_details=True)
```

### 5. Visualization
```python
from turingdb_kgsearch.visualization import visualize_graph_with_pyvis

# Interactive HTML visualization
visualize_graph_with_pyvis(
    subgraph,
    output_file='graph.html'
)
```

### 6. Workflow
```python
from turingdb_kgsearch.workflow import search_and_expand_hybrid_filtered, generate_report_hybrid_workflow_results

query = "What are the key privacy requirements?"
print(f"Query: '{query}'")

semantic_results, expanded, subgraph = search_and_expand_hybrid_filtered(
    query=query,
    G=G,
    node_vectors=node_vectors_context_heavy,
    node_texts=node_texts,
    sparse_vectors=node_vectors_sparse,
    sparse_vectorizer=vectorizer_sparse,
    structural_vectors=node_vectors_node2vec,
    model=model,
    k_search=5,
    max_hops=10,
    min_structural_sim=0.1,  # Must be structurally similar
    min_semantic_sim=0.1,  # AND semantically relevant to query
    structural_weight=0.5,  # 50-50 balance
    alpha=0.7,  # Weight alpha to attribute to semantic (dense) search, (1 - alpha) for keyword (sparse) search
)

report = generate_report_hybrid_workflow_results(semantic_results, expanded)
print(report)
```

### 6. LLM Integration

FYI: You will need to use your own LLM credentials for the LLM integration.
Advise: Save your credentials in `.env` file and run:
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_keys = {
    "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "Mistral": os.getenv("MISTRAL_API_KEY"),
    "OpenAI": os.getenv("OPENAI_API_KEY"),
}
```

Create prompt and query LLM:
```python
from turingdb_kgsearch.llm import graph_to_llm_context, create_llm_prompt_with_graph, query_llm
from IPython.display import display, Markdown

# Convert graph to LLM-friendly format
graph_text = graph_to_llm_context(subgraph, format='natural')

# Create complete prompt
print(f"Query: '{query}'")
prompt = create_llm_prompt_with_graph(
    query=query,
    subgraph=subgraph,
    report=report,
    format='natural'
)

# Send to your LLM
provider = "Anthropic"

result = query_llm(
    prompt=prompt,
    provider=provider,
    api_key=api_keys[provider],
    temperature=0.2,
)
display(Markdown(result))
```

## Example: Regulatory Compliance

Map controls across standards (ISO, NIST, EU AI Act, SOC2):
```python
# Build graph: Domain -> Topic -> Control -> Standards
# Search for relevant controls
results = hybrid_search("data encryption requirements", ...)

# Expand to related controls and standards
subgraph = search_and_expand_hybrid_filtered(
    query="data encryption requirements",
    k_search=2,
    max_hops=2,
    min_semantic_sim=0.6  # Stay on-topic
)

# Analyze cross-standard overlap
stats = get_subgraph_stats(subgraph)
visualize_graph_pyvis(subgraph)
```

See `notebooks/ai_governance_example.ipynb` for complete example.

## When to Use KGSearch

✅ **Good for:**
- Hierarchical content with relationships
- Cross-referenced documents (standards, citations)
- Knowledge bases with explicit structure
- Multi-aspect queries requiring context

❌ **Overkill for:**
- Flat document collections
- Simple keyword search

## Requirements

- Python ≥3.13
- NetworkX, sentence-transformers, scikit-learn, numpy, pandas

## Development
```bash
git clone https://github.com/turing-db/turingdb-kgsearch.git
cd turingdb-kgsearch
uv sync
uv run pytest
uv run ruff check
```

## License

MIT

## Citation
```bibtex
@software{turingdb_kgsearch,
  title = {TuringDB KGSearch: Knowledge Graph Search with Vector Embeddings},
  year = {2025},
  url = {https://github.com/turing-db/turingdb-kgsearch}
}
```