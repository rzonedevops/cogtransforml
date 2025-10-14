# OpenCog HypergraphQL Transformer Model

## Overview

The HypergraphQL model is a transformer-based architecture designed for processing and querying knowledge graphs represented as hypergraphs. It extends traditional transformer models with hypergraph-aware attention mechanisms and graph convolution layers.

## Architecture

### Key Components

1. **Hypergraph Attention**: A specialized attention mechanism that processes hyperedges (edges connecting multiple nodes) in addition to standard self-attention.

2. **Graph Convolution Layers**: Message passing layers that propagate information across the hypergraph structure.

3. **Node and Hyperedge Embeddings**: Separate embedding spaces for nodes and hyperedges to capture structural information.

4. **Standard Transformer Layers**: Traditional multi-head self-attention and feed-forward networks.

## Model Structure

### Hyperparameters

- `n_vocab`: Vocabulary size (default: 50257)
- `n_ctx`: Context length (default: 2048)
- `n_embd`: Embedding dimension (default: 768)
- `n_head`: Number of attention heads (default: 12)
- `n_layer`: Number of transformer layers (default: 12)
- `n_hyperedge`: Maximum nodes per hyperedge (default: 4)
- `n_graph_layers`: Number of graph convolution layers (default: 3)
- `n_relation_types`: Number of relation types for multi-relational support (default: 16) *[Phase 2]*

### Layer Architecture

Each layer consists of:
1. Layer normalization
2. Hypergraph-aware multi-head attention
3. Relation-aware attention *[Phase 2]*
4. Graph convolution operation
5. Relation-aware graph convolution *[Phase 2]*
6. Feed-forward network
7. Residual connections

## Usage

```python
from ctransformers import AutoModelForCausalLM

# Load a HypergraphQL model
llm = AutoModelForCausalLM.from_pretrained(
    "/path/to/hypergraphql-model.bin",
    model_type="hypergraphql"
)

# Generate text
print(llm("Query: What is OpenCog?"))

# Get embeddings (requires model implementation of Embeddings() method)
embeddings = llm.embed("OpenCog is a framework for AGI")
```

## Use Cases

1. **Knowledge Graph Querying**: Process complex queries over knowledge graphs
2. **Relational Reasoning**: Perform multi-hop reasoning over graph structures
3. **Semantic Search**: Find semantically similar concepts in hypergraph representations
4. **Graph-to-Text Generation**: Generate natural language descriptions from graph structures

## Technical Details

### Hypergraph Representation

- **Nodes**: Represent entities or concepts
- **Hyperedges**: Connect multiple nodes simultaneously (unlike traditional edges that connect only two nodes)
- **Attributes**: Both nodes and hyperedges can have associated attributes

### Attention Mechanism

The hypergraph attention mechanism extends standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

With hypergraph structure awareness:

```
HypergraphAttention(Q, K, V, E) = W_e * softmax(QK^T / sqrt(d_k)) V
```

Where `E` represents hyperedge structure and `W_e` are learned hyperedge attention weights.

### Graph Convolution

Message passing over hypergraph structure:

```
h_v^(l+1) = σ(W^(l) * AGG({h_u^(l) : u ∈ N(v)}))
```

Where `N(v)` is the neighborhood of node `v` in the hypergraph.

### Phase 2: Multi-Relational Support

The Phase 2 enhancements add support for typed relationships in hypergraphs:

#### Relation Type Embeddings

Each relation type (e.g., "is-a", "part-of", "causes") has a learned embedding:

```
R_emb: {relation_type_id} → ℝ^d
```

#### Relation-Aware Attention

The attention mechanism is extended to consider relation types:

```
RelationAttention(Q, K, V, R) = W_r(R) ⊙ Attention(Q, K, V)
```

Where `W_r(R)` is a relation-specific weighting computed from relation embeddings.

#### Relation-Aware Graph Convolution

Message passing considers relation types:

```
h_v^(l+1) = σ(W_r(R) ⊙ W^(l) * AGG({h_u^(l) : u ∈ N_r(v)}))
```

Where `N_r(v)` is the neighborhood of `v` connected by relation type `r`.

#### Common Relation Types

The model supports configurable relation types, commonly including:
- **is-a**: Taxonomic relationships (inheritance)
- **part-of**: Compositional relationships (meronymy)
- **causes**: Causal relationships
- **located-at**: Spatial relationships
- **temporal**: Temporal relationships
- **similar-to**: Similarity relationships
- **opposite-of**: Antonym relationships
- **custom**: User-defined relation types

## Training

To train a HypergraphQL model:

1. Prepare your hypergraph dataset in the appropriate format
2. Convert to GGML format with hypergraph structure annotations
3. Use standard language model training objectives with additional graph-aware losses

## Model Format

HypergraphQL models use the GGML format with additional tensors for:
- Node embeddings
- Hyperedge embeddings
- Hyperedge attention weights
- Graph convolution weights

## Phase 2 Enhancements

- [x] Multi-relational hyperedge types
- [x] Relation type embeddings
- [x] Relation-aware attention mechanism
- [x] Relation-aware graph convolution
- [x] Dynamic relation type support

## Phase 3 Enhancements (Current)

- [x] OpenCog AtomSpace integration
- [x] Temporal hypergraph evolution
- [x] Dynamic hypergraph structures (runtime modification)
- [x] Hierarchical relation types
- [x] Context-based relation inference
- [x] Bidirectional AtomSpace synchronization
- [x] Time-aware embeddings and attention

### OpenCog AtomSpace Integration

Connect directly to OpenCog's AtomSpace:

```python
llm = AutoModelForCausalLM.from_pretrained(
    "model.bin",
    model_type="hypergraphql",
    atomspace_uri="atomspace://localhost:5000"
)

# Query atoms
result = llm.query_atoms(
    pattern="(InheritanceLink ?x (ConceptNode 'Animal'))"
)

# Enable sync
llm.enable_atomspace_sync(read=True, write=True)
```

### Temporal Reasoning

Reason over time-varying knowledge:

```python
# Query historical state
response = llm(
    "What was the relationship in 1980?",
    temporal_context={"year": 1980}
)

# Track evolution
evolution = llm.track_evolution(
    entity="AI",
    time_range=("2010-01-01", "2025-01-01")
)
```

### Dynamic Graph Modification

Modify graph structure during inference:

```python
# Add nodes and edges
llm.add_node("NewConcept", "ConceptNode", {"importance": 0.9})
llm.add_edge("NewConcept", "Existing", "is-a", confidence=0.95)

# Remove edges
llm.remove_edge(edge_id="old_connection")
```

### Hierarchical Relations

Organize relation types in hierarchies:

```python
llm.set_relation_hierarchy({
    "Physical": {
        "part-of": {"weight": 1.0},
        "connected-to": {"weight": 0.8}
    },
    "Conceptual": {
        "is-a": {"weight": 1.0},
        "similar-to": {"weight": 0.7}
    }
})
```

### Relation Inference

Automatically infer relation types:

```python
result = llm.infer_relation(
    source="heart",
    target="body",
    context="The heart is in the body",
    return_confidence=True
)
# Returns: {"relation": "part-of", "confidence": 0.89}
```

## Future Enhancements (Phase 4+)

- [ ] SPARQL-like query language integration
- [ ] CUDA/Metal acceleration for hypergraph operations
- [ ] Large-scale graph optimization
- [ ] Distributed inference support

## References

- OpenCog Framework: https://opencog.org/
- Hypergraph Neural Networks
- Graph Attention Networks (GAT)
- Transformer Architecture

## License

This implementation is part of the CTransformers project and follows the same MIT license.
