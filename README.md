# TuringDB GraphRAG

A Python library for semantic search and graph-based retrieval augmented generation (RAG) over knowledge graphs. Combines dense vector search, sparse keyword search, and graph structure exploration using Node2Vec.

## Overview

TuringDB GraphRAG enables you to:
- **Search graphs semantically**: Find nodes using natural language queries with hybrid (semantic + keyword) search
- **Explore graph structure**: Traverse graph neighborhoods filtered by both structural and semantic similarity
- **Generate context for LLMs**: Export graph data in LLM-friendly formats for retrieval-augmented generation
- **Visualize results**: Create interactive graph visualizations with PyVis

## How This Differs from Microsoft's GraphRAG

While both approaches use graphs for retrieval-augmented generation, they solve different problems:

### Microsoft's GraphRAG
- **Use case**: Extract structure from unstructured text
- **Approach**: LLM generates entity graph → builds community summaries → answers via hierarchical summarization
- **Best for**: Large text corpora without existing structure (news articles, research papers, documents)
- **Pre-computation**: Heavy (entity extraction + multi-level summaries)

**Example**: "What are the main themes in 1M news articles?"

### TuringDB GraphRAG
- **Use case**: Query existing knowledge graphs with semantic awareness
- **Approach**: Hybrid search (semantic + keywords) → graph traversal → filter by structure + semantics → retrieve subgraph
- **Best for**: Structured knowledge graphs (regulatory standards, ontologies, citation networks, knowledge bases)
- **Pre-computation**: Light (embeddings only)

**Example**: "Find data privacy controls and related requirements across ISO/NIST standards"

### Key Differences

| Aspect | Microsoft GraphRAG | TuringDB GraphRAG |
|--------|-------------------|-------------------|
| **Input** | Unstructured text | Existing knowledge graph |
| **Graph creation** | LLM-generated | Pre-existing |
| **Core technique** | Community detection + summarization | Hybrid search + Node2Vec filtering |
| **Retrieval** | Pre-computed summaries | Dynamic subgraph extraction |
| **Query types** | Global themes + local facts | Semantic + structural exploration |

### When to Use Each

**Use Microsoft's GraphRAG when**:
- You have large unstructured text collections
- You need to discover themes and patterns
- You want to answer "global" questions about the corpus

**Use TuringDB GraphRAG when**:
- You already have a knowledge graph
- You need precise semantic + structural retrieval
- You want to explore graph neighborhoods with relevance filtering
- You need to constrain LLM context to specific subgraphs

### Novel Contributions

TuringDB GraphRAG introduces:
1. **Hybrid filtering**: Combines structural similarity (Node2Vec) with semantic relevance during graph traversal
2. **Semantic drift prevention**: Ensures exploration stays on-topic via continuous query similarity checks
3. **Multi-modal search**: Dense (semantic) + sparse (keyword) + structural (graph topology)
4. **Flexible embeddings**: Node-only, context-enriched, or smart aggregation strategies

## Key Features

- **Multiple Search Methods**:
  - Dense search (semantic similarity using sentence transformers)
  - Sparse search (keyword matching using TF-IDF/BM25)
  - Hybrid search (combines both with configurable weighting)

- **Graph Structure Awareness**:
  - Node2Vec embeddings capture graph topology
  - Filter graph traversal by structural similarity
  - Combine semantic relevance with graph structure

- **Flexible Text Extraction**:
  - Node-only embeddings (basic)
  - Context-enriched embeddings (includes neighbors)
  - Smart enriched embeddings (type-aware context)

- **LLM Integration**:
  - Export graphs in multiple formats (Natural language, Markdown, JSON, Cypher)
  - Generate complete prompts with graph context
  - Constrain LLM responses to subgraph data only

## Installation

Using uv:
```bash
uv add turingdb-graphrag
```

Using pip:
```bash
pip install turingdb-graphrag
```

### Dependencies

```
networkx
numpy
scikit-learn
sentence-transformers
node2vec
pyvis
```

## Quick Start

### 1. Basic Setup

```python
import networkx as nx
from sentence_transformers import SentenceTransformer
from turingdb_graphrag import (
    build_node_only_embeddings,
    build_sparse_embeddings,
    hybrid_search
)

# Load your graph
G = nx.Graph()
# ... add nodes and edges ...

# Load embedding model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Build embeddings
node_vectors, node_texts = build_node_only_embeddings(G, model)
sparse_vectors, sparse_vectorizer = build_sparse_embeddings(G)
```

### 2. Search Your Graph

```python
# Hybrid search (semantic + keywords)
results = hybrid_search(
    query="data privacy protection",
    node_vectors=node_vectors,
    node_texts=node_texts,
    sparse_vectors=sparse_vectors,
    sparse_vectorizer=sparse_vectorizer,
    model=model,
    k=5,
    alpha=0.7,  # 70% semantic, 30% keywords
    node_type='control'
)

# Print results
for result in results:
    print(f"{result['node_id']}: {result['similarity']:.3f}")
    print(f"  {result['text'][:100]}...")
```

### 3. Explore Graph Structure

```python
from turingdb_graphrag import (
    build_node2vec_embeddings,
    search_and_expand_hybrid_filtered
)

# Build structural embeddings
structural_vectors = build_node2vec_embeddings(G, dimensions=128)

# Two-stage workflow: semantic search + structural expansion
semantic_results, expanded, subgraph = search_and_expand_hybrid_filtered(
    query="data privacy protection",
    G=G,
    node_vectors=node_vectors,
    node_texts=node_texts,
    sparse_vectors=sparse_vectors,
    sparse_vectorizer=sparse_vectorizer,
    structural_vectors=structural_vectors,
    model=model,
    k_search=2,              # Find 2 seed nodes
    max_hops=2,              # Traverse up to 2 hops
    min_structural_sim=0.7,  # Structural similarity threshold
    min_semantic_sim=0.6,    # Semantic similarity threshold
    structural_weight=0.5    # Balance structure vs semantics
)

# Subgraph contains all original data + similarity scores
print(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
```

### 4. Generate LLM Context

```python
from turingdb_graphrag import create_llm_prompt_with_graph

# Get report
report = print_hybrid_workflow_results(semantic_results, expanded, subgraph)

# Create LLM prompt
prompt = create_llm_prompt_with_graph(
    query="data privacy protection",
    subgraph=subgraph,
    report=report,
    format='natural'  # or 'markdown', 'json', 'cypher'
)

# Send to your LLM
response = llm.query(prompt)
```

### 5. Visualize Results

```python
from turingdb_graphrag import visualize_graph_pyvis

# Create interactive visualization
visualize_graph_pyvis(
    subgraph,
    output_file='graph.html',
    node_color_attr='type',
    node_size_attr='combined_score'
)
# Opens graph.html in browser
```

## Core Concepts

### Embedding Types

#### Dense Embeddings (Semantic)
Uses sentence transformers to capture semantic meaning:

```python
# Option 1: Node-only (basic)
node_vectors, node_texts = build_node_only_embeddings(G, model)

# Option 2: Context-enriched (includes neighbor text)
node_vectors, node_texts = build_context_enriched_embeddings(G, model)

# Option 3: Smart enriched (type-aware context aggregation)
node_vectors, node_texts = build_smart_enriched_embeddings(G, model)
```

#### Sparse Embeddings (Keywords)
Uses TF-IDF for keyword matching:

```python
sparse_vectors, sparse_vectorizer = build_sparse_embeddings(
    G,
    max_features=500,
    ngram_range=(1, 2)
)
```

#### Structural Embeddings (Graph Topology)
Uses Node2Vec to capture graph structure:

```python
structural_vectors = build_node2vec_embeddings(
    G,
    dimensions=128,
    walk_length=10,
    num_walks=80
)
```

### Search Methods

#### Dense Search
Pure semantic similarity:

```python
results = dense_search(
    query="privacy controls",
    node_vectors=node_vectors,
    node_texts=node_texts,
    G=G,
    model=model,
    k=5
)
```

#### Sparse Search
Keyword-based matching:

```python
results = sparse_search(
    query="ISO27001 controls",
    sparse_vectors=sparse_vectors,
    sparse_vectorizer=sparse_vectorizer,
    node_texts=node_texts,
    k=5
)
```

#### Hybrid Search
Combines semantic + keywords:

```python
results = hybrid_search(
    query="NIST risk management",
    node_vectors=node_vectors,
    node_texts=node_texts,
    sparse_vectors=sparse_vectors,
    sparse_vectorizer=sparse_vectorizer,
    model=model,
    k=5,
    alpha=0.7  # 0.0=sparse only, 1.0=dense only, 0.7=balanced
)
```

### Workflow: Search + Expand

The `search_and_expand_hybrid_filtered` function implements a two-stage workflow:

**Stage 1: Semantic Search**
- Uses hybrid search (dense + sparse) to find seed nodes matching the query
- Returns top-k most relevant nodes

**Stage 2: Hybrid-Filtered Expansion**
- Traverses graph from seed nodes (up to `max_hops` distance)
- Filters neighbors by BOTH:
  - Structural similarity (Node2Vec) ≥ `min_structural_sim`
  - Semantic similarity to query ≥ `min_semantic_sim`
- Prevents semantic drift while exploring graph structure

**Returns**:
- `semantic_results`: Seed nodes from initial search
- `expanded`: Detailed expansion info for each seed
- `subgraph`: NetworkX graph with all found nodes + similarity scores

## Advanced Usage

### Custom Text Extraction

Control how node text is extracted for embeddings:

```python
def custom_extract(node_id, data):
    """Extract text from node for embedding."""
    if data.get('type') == 'control':
        return f"{data['domain']}: {data['statement']}"
    elif data.get('type') == 'topic':
        return f"Topic: {data['name']}"
    return str(node_id)

node_vectors, node_texts = build_node_only_embeddings(
    G,
    model,
    text_extractor=custom_extract
)
```

### Compare Search Methods

```python
from turingdb_graphrag import compare_search_methods

compare_search_methods(
    query="risk assessment",
    node_vectors=node_vectors,
    node_texts=node_texts,
    sparse_vectors=sparse_vectors,
    sparse_vectorizer=sparse_vectorizer,
    model=model,
    k=3
)
```

Output shows results from dense, sparse, and hybrid search side-by-side.

### Export Graphs for LLMs

Multiple format options:

```python
from turingdb_graphrag import graph_to_llm_context

# Natural language (most readable)
text = graph_to_llm_context(subgraph, format='natural')

# Markdown (structured)
text = graph_to_llm_context(subgraph, format='markdown')

# JSON (programmatic)
text = graph_to_llm_context(subgraph, format='json')

# Cypher (for graph databases)
text = graph_to_llm_context(subgraph, format='cypher')
```

### Subgraph Node Attributes

After `search_and_expand_hybrid_filtered`, nodes have:

**Seed nodes**:
- `is_seed`: True
- `seed_score`: Initial search score
- `dense_score`: Semantic component
- `sparse_score`: Keyword component

**Found neighbors**:
- `is_seed`: False
- `seed_node`: Which seed node led to this
- `hop_distance`: Graph distance from seed
- `direction`: 'predecessor' or 'successor'
- `structural_similarity`: Node2Vec similarity
- `semantic_similarity`: Semantic similarity to query
- `combined_score`: Weighted combination

**Intermediate nodes**:
- `is_intermediate`: True (on path between found nodes)

All nodes retain their original attributes.

## Use Cases

### 1. Regulatory Compliance Mapping
Search controls across standards (ISO, NIST, SOC2) and find related requirements:

```python
results, expanded, subgraph = search_and_expand_hybrid_filtered(
    query="data encryption requirements",
    # ... parameters ...
)
# Subgraph contains related controls and their standard mappings
```

### 2. Knowledge Graph Exploration
Start with a concept and explore semantically similar + structurally related concepts:

```python
results = hybrid_search(query="machine learning", ...)
subgraph = search_and_expand_hybrid_filtered(
    query="machine learning",
    min_semantic_sim=0.6  # Stay on-topic
)
```

### 3. RAG for Graph Data
Generate LLM context constrained to relevant subgraph:

```python
prompt = create_llm_prompt_with_graph(
    query="What are the privacy requirements?",
    subgraph=subgraph,
    report=report,
    format='natural'
)
# LLM answers using only subgraph data
```

### 4. Citation Networks
Find related papers and their citation neighborhoods:

```python
# Nodes = papers, edges = citations
results, expanded, subgraph = search_and_expand_hybrid_filtered(
    query="graph neural networks",
    min_structural_sim=0.7  # Papers with similar citation patterns
)
```

## API Reference

### Embedding Functions

**`build_node_only_embeddings(G, model, text_extractor=None)`**
- Builds embeddings from node text only
- Returns: `(node_vectors, node_texts)`

**`build_context_enriched_embeddings(G, model, max_neighbors=5)`**
- Includes neighbor text in embeddings
- Returns: `(node_vectors, node_texts)`

**`build_smart_enriched_embeddings(G, model, max_neighbors=5)`**
- Type-aware neighbor aggregation
- Returns: `(node_vectors, node_texts)`

**`build_sparse_embeddings(G, max_features=500, ngram_range=(1,2))`**
- TF-IDF sparse vectors
- Returns: `(sparse_vectors, sparse_vectorizer)`

**`build_node2vec_embeddings(G, dimensions=128, walk_length=10, num_walks=80)`**
- Structural embeddings via random walks
- Returns: `structural_vectors`

### Search Functions

**`dense_search(query, node_vectors, node_texts, G, model, k=5, node_type=None)`**
- Semantic search only
- Returns: List of results with similarity scores

**`sparse_search(query, sparse_vectors, sparse_vectorizer, node_texts, k=5, node_type=None)`**
- Keyword search only
- Returns: List of results with similarity scores

**`hybrid_search(query, node_vectors, node_texts, sparse_vectors, sparse_vectorizer, model, k=5, alpha=0.7, node_type=None)`**
- Combined semantic + keyword search
- `alpha`: 0.0=sparse, 1.0=dense, 0.7=balanced
- Returns: List of results with dense/sparse/combined scores

### Workflow Functions

**`search_and_expand_hybrid_filtered(...)`**
- Two-stage: search + filtered expansion
- Parameters:
  - `k_search`: Number of seed nodes
  - `max_hops`: Maximum graph distance
  - `min_structural_sim`: Node2Vec threshold
  - `min_semantic_sim`: Semantic threshold
  - `structural_weight`: Balance structure/semantics
- Returns: `(semantic_results, expanded, subgraph)`

**`print_hybrid_workflow_results(semantic_results, expanded, subgraph=None)`**
- Generates text report
- Returns: String (for saving or LLM input)

### LLM Integration

**`graph_to_llm_context(G, format='natural', include_attributes=True, max_nodes=None)`**
- Converts graph to LLM-friendly text
- Formats: 'natural', 'markdown', 'json', 'cypher'
- Returns: String

**`create_llm_prompt_with_graph(query, subgraph, report, format='natural', custom_task=None)`**
- Creates complete LLM prompt
- Returns: Formatted prompt string

### Visualization

**`visualize_graph_pyvis(G, output_file='graph.html', node_color_attr='type', node_label_attr=None, node_size_attr=None)`**
- Creates interactive HTML visualization
- Returns: IFrame (if in Jupyter) or None

## Performance Tips

1. **Model Selection**:
   - `paraphrase-MiniLM-L3-v2`: Fast, good for general use
   - `all-MiniLM-L6-v2`: Balanced speed/quality
   - `all-mpnet-base-v2`: Best quality, slower

2. **Sparse Vector Optimization**:
   - Reduce `max_features` for faster search
   - Use `ngram_range=(1,1)` for exact word matching only

3. **Node2Vec Parameters**:
   - Lower `num_walks` for faster training
   - Lower `dimensions` for less memory

4. **Subgraph Size**:
   - Limit `max_hops` and `k_search` to control subgraph size
   - Higher thresholds = smaller, more relevant subgraphs

## Limitations

- **Node2Vec non-deterministic**: Results vary between runs even with fixed seed
- **Memory intensive**: HNSW and Node2Vec require vectors in memory
- **Not suitable for**:
  - Streaming/dynamic graphs (requires rebuild)
  - Very large graphs (>1M nodes without optimization)
  - Real-time applications (indexing takes time)

## Examples

See the `examples/` directory for complete notebooks:
- `regulatory_compliance.ipynb`: ISO/NIST standards mapping
- `citation_network.ipynb`: Academic paper exploration
- `knowledge_graph.ipynb`: General knowledge graph RAG

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file

## Citation

If you use TuringDB GraphRAG in research, please cite:

```bibtex
@software{turingdb_graphrag,
  title = {TuringDB GraphRAG: Semantic Search and RAG for Knowledge Graphs},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/turingdb-graphrag}
}
```

## Related Work

- **Node2Vec**: Grover & Leskovec (2016) - "node2vec: Scalable Feature Learning for Networks"
- **Sentence Transformers**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Graph Neural Networks**: Various GNN architectures for knowledge graphs

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/turingdb-graphrag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/turingdb-graphrag/discussions)
- **Email**: your.email@example.com
