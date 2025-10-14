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

### Layer Architecture

Each layer consists of:
1. Layer normalization
2. Hypergraph-aware multi-head attention
3. Graph convolution operation
4. Feed-forward network
5. Residual connections

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

## Future Enhancements

- [ ] Support for dynamic hypergraph structures
- [ ] Integration with OpenCog AtomSpace
- [ ] Multi-relational hyperedge types
- [ ] Temporal hypergraph evolution
- [ ] Integration with SPARQL-like query languages

## References

- OpenCog Framework: https://opencog.org/
- Hypergraph Neural Networks
- Graph Attention Networks (GAT)
- Transformer Architecture

## License

This implementation is part of the CTransformers project and follows the same MIT license.
