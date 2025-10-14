# OpenCog HypergraphQL Transformer Documentation

## Overview

This directory contains comprehensive documentation for the OpenCog HypergraphQL Transformer implementation in the CTransformers framework.

## Documentation Files

### Core Documentation

- **[HYPERGRAPHQL.md](HYPERGRAPHQL.md)** - Main technical documentation
  - Architecture overview
  - Model structure and hyperparameters
  - Usage examples
  - Training instructions
  - Technical details on attention and graph convolution

### Phase-Specific Documentation

- **[PHASE2_MULTI_RELATIONAL.md](PHASE2_MULTI_RELATIONAL.md)** - Phase 2 implementation details
  - Multi-relational hypergraph support
  - Relation type embeddings
  - Relation-aware attention and convolution
  - Usage examples with typed relationships
  - Performance considerations

- **[PHASE3_ATOMSPACE.md](PHASE3_ATOMSPACE.md)** - Phase 3 implementation details
  - OpenCog AtomSpace integration
  - Temporal hypergraph evolution
  - Dynamic graph structure modification
  - Hierarchical relation types
  - Context-based relation inference
  - Advanced AGI features

- **[PHASE3_VISUAL_GUIDE.md](PHASE3_VISUAL_GUIDE.md)** - Phase 3 visual architecture
  - Component diagrams
  - Data flow visualizations
  - Memory layout
  - Integration patterns

- **[PHASE4_PERFORMANCE.md](PHASE4_PERFORMANCE.md)** - Phase 4 implementation details (🚧 In Progress)
  - HyperQL query language specification
  - GPU acceleration (CUDA/Metal)
  - Large-scale graph optimization
  - Distributed inference architecture
  - Production deployment tools
  - Performance benchmarks

- **[PHASE4_VISUAL_GUIDE.md](PHASE4_VISUAL_GUIDE.md)** - Phase 4 visual architecture (🚧 In Progress)
  - System architecture layers
  - Query processing pipeline
  - GPU acceleration diagrams
  - Distributed architecture
  - Production deployment patterns

## Implementation Phases

### Phase 1 ✅ Complete
Core hypergraph transformer with:
- Hypergraph-aware attention mechanism
- Graph convolution layers
- Node and hyperedge embeddings
- Basic model architecture

**Documentation**: See [HYPERGRAPHQL.md](HYPERGRAPHQL.md) Sections 1-5

### Phase 2 ✅ Complete
Multi-relational support with:
- Relation type embeddings (16 types by default)
- Relation-aware attention
- Relation-aware graph convolution
- Support for typed relationships (is-a, part-of, causes, etc.)

**Documentation**: See [PHASE2_MULTI_RELATIONAL.md](PHASE2_MULTI_RELATIONAL.md)

### Phase 3 ✅ Complete
Advanced features:
- OpenCog AtomSpace integration
- Temporal hypergraph evolution
- Dynamic graph structure modification at runtime
- Hierarchical relation types
- Context-based relation inference

**Documentation**: See [PHASE3_ATOMSPACE.md](PHASE3_ATOMSPACE.md) and [PHASE3_VISUAL_GUIDE.md](PHASE3_VISUAL_GUIDE.md)

### Phase 4 🚧 In Progress
Performance and scaling:
- HyperQL query language (SPARQL-like for hypergraphs)
- CUDA/Metal GPU acceleration (3-5x speedup)
- Large-scale graph optimization (billion-edge support)
- Distributed inference (multi-node clusters)
- Production deployment tools

**Documentation**: See [PHASE4_PERFORMANCE.md](PHASE4_PERFORMANCE.md) and [PHASE4_VISUAL_GUIDE.md](PHASE4_VISUAL_GUIDE.md)

## Quick Start

### Installation

```bash
pip install ctransformers
```

### Basic Usage

```python
from ctransformers import AutoModelForCausalLM

# Load HypergraphQL model
llm = AutoModelForCausalLM.from_pretrained(
    "path/to/hypergraphql-model.bin",
    model_type="hypergraphql"
)

# Generate text
response = llm("Query: What is OpenCog?", max_new_tokens=100)
print(response)
```

### Phase 2 Multi-Relational Usage

```python
# Query with relation types
query = """
Using 'is-a' and 'part-of' relations:
What is the relationship between neurons and the brain?
"""
response = llm(query, max_new_tokens=150)
print(response)
```

See [examples/hypergraphql_example.py](../examples/hypergraphql_example.py) for more examples.

## Model Hyperparameters

| Parameter | Default | Description | Phase |
|-----------|---------|-------------|-------|
| `n_vocab` | 50257 | Vocabulary size | 1 |
| `n_ctx` | 2048 | Context length | 1 |
| `n_embd` | 768 | Embedding dimension | 1 |
| `n_head` | 12 | Attention heads | 1 |
| `n_layer` | 12 | Transformer layers | 1 |
| `n_hyperedge` | 4 | Max nodes per hyperedge | 1 |
| `n_graph_layers` | 3 | Graph convolution layers | 1 |
| `n_relation_types` | 16 | Number of relation types | 2 |
| `n_temporal_steps` | 1000 | Temporal snapshots | 3 |
| `n_hierarchy_levels` | 4 | Relation hierarchy depth | 3 |
| `n_inference_dims` | 768 | Relation inference dimensions | 3 |

## Architecture Diagram

```
Input Tokens
     ↓
Token Embeddings + Position Embeddings
     ↓
[For each layer:]
     ↓
Layer Normalization
     ↓
Multi-Head Attention (Q, K, V)
     ↓
Hypergraph Attention
     ↓
Relation-Aware Attention [Phase 2]
     ↓
Temporal & Hierarchical Attention [Phase 3]
     ↓
Residual Connection
     ↓
Graph Convolution
     ↓
Relation-Aware Graph Convolution [Phase 2]
     ↓
Temporal Graph Convolution [Phase 3]
     ↓
Dynamic Graph Updates [Phase 3]
     ↓
Relation Inference [Phase 3]
     ↓
Layer Normalization
     ↓
Feed-Forward Network
     ↓
Residual Connection
     ↓
Output
```

## Key Features by Phase

### Phase 1 Features
- ✅ Hypergraph-aware attention
- ✅ Graph convolution layers
- ✅ Node and hyperedge embeddings
- ✅ Standard transformer components
- ✅ GGML integration

### Phase 2 Features
- ✅ Multi-relational support
- ✅ Relation type embeddings
- ✅ Relation-aware attention
- ✅ Relation-aware convolution
- ✅ Typed relationship processing

### Phase 3 Features
- ✅ OpenCog AtomSpace integration
- ✅ Temporal hypergraph evolution
- ✅ Dynamic graph modification
- ✅ Hierarchical relation types
- ✅ Context-based relation inference
- ✅ Bidirectional AtomSpace sync
- ✅ Time-aware embeddings

## Use Cases

1. **Knowledge Graph Querying** - Query complex knowledge graphs using natural language
2. **Relational Reasoning** - Perform multi-hop reasoning over typed relationships
3. **Semantic Search** - Find semantically similar concepts in hypergraphs
4. **Graph-to-Text Generation** - Generate natural language from graph structures
5. **OpenCog Integration** [Phase 3] - Direct interface with OpenCog's AtomSpace for AGI applications
6. **Multi-Relational Reasoning** [Phase 2] - Reason over typed relationships (is-a, part-of, causes)
7. **Temporal Reasoning** [Phase 3] - Track and reason about time-varying knowledge
8. **Dynamic Graphs** [Phase 3] - Modify graph structure during inference
9. **Hierarchical Relations** [Phase 3] - Multi-level relation type organization
10. **Smart Inference** [Phase 3] - Automatically infer relation types from context

## Testing

Tests are located in `tests/test_hypergraphql.py`:

```bash
# Run all tests
pytest tests/test_hypergraphql.py

# Run specific test class
pytest tests/test_hypergraphql.py::TestHypergraphQL
pytest tests/test_hypergraphql.py::TestHypergraphQLPhase2
pytest tests/test_hypergraphql.py::TestHypergraphQLPhase3
```

## Examples

See the `examples/` directory for complete working examples:

- **hypergraphql_example.py** - Comprehensive examples covering all features
  - Knowledge graph querying
  - Multi-relational queries (Phase 2)
  - Temporal reasoning (Phase 3)
  - Dynamic graph modification (Phase 3)
  - AtomSpace integration (Phase 3)
  - Relational reasoning
  - Graph embeddings
  - Streaming generation

## References

### Academic Papers
- **Graph Attention Networks (GAT)** - Veličković et al., 2018
- **Relational Graph Convolutional Networks** - Schlichtkrull et al., 2018
- **Hypergraph Neural Networks** - Feng et al., 2019
- **Knowledge Graph Embeddings** - Bordes et al., 2013

### Related Projects
- **OpenCog Framework**: https://opencog.org/
- **GGML**: https://github.com/ggerganov/ggml
- **CTransformers**: https://github.com/marella/ctransformers

## Contributing

When contributing to the HypergraphQL implementation:

1. Maintain backward compatibility with previous phases
2. Follow existing code patterns in `models/llms/`
3. Update relevant documentation files
4. Add tests for new features
5. Update examples to demonstrate new capabilities

## License

This implementation is part of the CTransformers project and follows the same MIT license.

## Contact

For questions, issues, or contributions related to HypergraphQL:
- GitHub Issues: https://github.com/rzonedevops/cogtransforml/issues
- Main Repository: https://github.com/rzonedevops/cogtransforml

## Version History

### v0.3.0 (Phase 3) - Current
- OpenCog AtomSpace integration
- Temporal hypergraph evolution
- Dynamic graph modification
- Hierarchical relation types
- Context-based relation inference
- Advanced AGI features

### v0.2.0 (Phase 2)
- Added multi-relational support
- Relation type embeddings
- Relation-aware attention and convolution
- Extended documentation and examples

### v0.1.0 (Phase 1)
- Initial HypergraphQL implementation
- Core hypergraph attention
- Graph convolution layers
- Basic model architecture

---

**Last Updated**: October 2025  
**Current Phase**: Phase 3 - OpenCog AtomSpace Integration
