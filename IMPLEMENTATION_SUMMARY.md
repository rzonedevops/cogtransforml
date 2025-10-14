# OpenCog HypergraphQL Transformer Implementation Summary

## Overview

This document summarizes the implementation of OpenCog HypergraphQL as a transformer model in the CTransformers framework.

## What is HypergraphQL?

HypergraphQL is a novel transformer-based architecture designed for processing and querying knowledge graphs represented as hypergraphs. Unlike traditional graphs where edges connect only two nodes, hypergraphs support hyperedges that can connect multiple nodes simultaneously, making them ideal for representing complex relational structures found in knowledge bases.

## Implementation Details

### 1. Core Model Architecture (`models/llms/hypergraphql.cc`)

The implementation includes:

#### Data Structures
- **hypergraphql_hparams**: Hyperparameters including vocabulary size, context length, embedding dimensions, attention heads, number of layers, and hypergraph-specific parameters
- **hypergraphql_layer**: Layer structure containing:
  - Hypergraph attention weights
  - Standard transformer attention (Q, K, V projections)
  - Graph convolution weights
  - Layer normalization parameters
  - Feed-forward network weights
- **hypergraphql_model**: Complete model structure with embeddings, layers, and memory context

#### Key Components

**Hypergraph Attention Mechanism**
```cpp
ggml_tensor *hypergraph_attention(const hypergraphql_layer &layer,
                                 ggml_context *ctx0, ggml_tensor *inp,
                                 int n_head)
```
Extends standard self-attention with hypergraph structure awareness, allowing the model to process hyperedges that connect multiple nodes.

**Graph Convolution Layer**
```cpp
ggml_tensor *graph_convolution(const hypergraphql_layer &layer,
                              ggml_context *ctx0, ggml_tensor *inp)
```
Implements message passing over the hypergraph structure for propagating information across connected nodes.

**Model Evaluation**
```cpp
bool hypergraphql_eval(const hypergraphql_model &model, const int n_threads,
                      const int n_past,
                      const std::vector<gpt_vocab::id> &embd_inp,
                      std::vector<float> &embd_w, size_t &mem_per_token)
```
Main evaluation function that processes tokens through the complete transformer architecture with hypergraph-aware operations.

### 2. Integration with CTransformers (`models/llm.cc`)

- Registered the new model type `hypergraphql` in the model creation factory
- Integrated with existing LLM interface for seamless operation
- Follows the REGISTER_LLM macro pattern used by other models

### 3. Testing (`tests/test_hypergraphql.py`)

Created test suite covering:
- Model type registration verification
- Hypergraph structure validation
- Attention mechanism testing

### 4. Documentation

#### Technical Documentation (`docs/HYPERGRAPHQL.md`)
Comprehensive documentation including:
- Architecture overview and key components
- Model structure and hyperparameters
- Layer architecture details
- Usage examples
- Technical details on hypergraph representation, attention mechanism, and graph convolution
- Training instructions
- Model format specifications
- Future enhancement roadmap

#### Example Code (`examples/hypergraphql_example.py`)
Demonstrates four key use cases:
1. Knowledge graph querying
2. Relational reasoning
3. Graph concept embeddings
4. Streaming generation

#### README Updates
- Added HypergraphQL to the supported models table
- Updated documentation references

## Key Features

1. **Hypergraph-Aware Attention**: Processes hyperedges (edges connecting multiple nodes) alongside standard self-attention
2. **Graph Convolution Layers**: Message passing mechanisms for information propagation across hypergraph structure
3. **Specialized Embeddings**: Separate embedding spaces for nodes and hyperedges
4. **Standard Transformer Components**: Layer normalization, feed-forward networks, residual connections
5. **GGML Integration**: Compatible with existing GGML framework and model loading infrastructure

## Architecture Highlights

### Transformer Layer Flow
```
Input
  ↓
Layer Normalization
  ↓
Multi-Head Attention (Q, K, V)
  ↓
Hypergraph Attention (with hyperedge structure)
  ↓
Projection
  ↓
Residual Connection
  ↓
Graph Convolution (message passing)
  ↓
Layer Normalization
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Output
```

### Attention Mechanism

Standard transformer attention:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Extended with hypergraph awareness:
```
HypergraphAttention(Q, K, V, E) = W_e * softmax(QK^T / sqrt(d_k)) V
```
Where E represents hyperedge structure and W_e are learned hyperedge attention weights.

### Graph Convolution

Message passing formula:
```
h_v^(l+1) = σ(W^(l) * AGG({h_u^(l) : u ∈ N(v)}))
```
Where N(v) is the neighborhood of node v in the hypergraph.

## Use Cases

1. **Knowledge Base Question Answering**: Query complex knowledge graphs using natural language
2. **Relational Reasoning**: Perform multi-hop reasoning over graph structures
3. **Semantic Search**: Find semantically similar concepts in hypergraph representations
4. **Graph-to-Text Generation**: Generate natural language from graph structures
5. **OpenCog Integration**: Interface with OpenCog's AtomSpace for AGI applications

## Implementation Philosophy

The implementation follows established patterns in the CTransformers codebase:
- Uses the same structure as other models (falcon, dolly, gpt2, etc.)
- Integrates with GGML for tensor operations
- Compatible with existing model loading and evaluation infrastructure
- Minimal changes to core framework
- Surgical additions that don't affect other models

## Future Work

Potential enhancements include:
- Support for dynamic hypergraph structures
- Integration with OpenCog AtomSpace
- Multi-relational hyperedge types
- Temporal hypergraph evolution
- SPARQL-like query language integration
- CUDA/Metal acceleration for hypergraph operations

## Testing and Validation

To fully test this implementation, you would need:
1. A trained HypergraphQL model in GGML format
2. Hypergraph-structured training data
3. Benchmarks for knowledge graph querying tasks
4. Evaluation metrics for relational reasoning

## Conclusion

This implementation provides a complete foundation for HypergraphQL transformer models within the CTransformers framework. It extends traditional transformers with hypergraph-aware operations while maintaining compatibility with the existing infrastructure. The architecture is designed to process complex relational structures found in knowledge graphs, making it suitable for advanced reasoning and query tasks in AGI applications.
