# HypergraphQL Examples

This directory contains example code demonstrating how to use the OpenCog HypergraphQL transformer model.

## hypergraphql_example.py

A comprehensive example showing various use cases for the HypergraphQL model:

1. **Knowledge Graph Querying** - Query and retrieve information from hypergraph-structured knowledge bases
2. **Relational Reasoning** - Perform multi-hop reasoning over graph structures
3. **Graph Concept Embeddings** - Generate embeddings for graph nodes and concepts
4. **Streaming Generation** - Stream responses for interactive applications

### Running the Example

```bash
python3 examples/hypergraphql_example.py
```

**Note:** These examples require a trained HypergraphQL model file. The code demonstrates the intended API usage even without a model file.

## Model Requirements

To run these examples with actual inference, you need:

1. A trained HypergraphQL model in GGML format
2. The model file should include:
   - Token embeddings
   - Position embeddings
   - Hypergraph node and edge embeddings
   - Transformer layer weights
   - Hypergraph attention weights
   - Graph convolution weights

## Use Cases

The HypergraphQL model is designed for:

- **Knowledge Base Question Answering**: Query complex knowledge graphs using natural language
- **Graph Neural Networks**: Process graph-structured data with transformer attention
- **Semantic Networks**: Navigate and reason over semantic relationships
- **OpenCog Integration**: Interface with OpenCog's AtomSpace for AGI applications

## Further Reading

See [docs/HYPERGRAPHQL.md](../docs/HYPERGRAPHQL.md) for detailed documentation on the model architecture and training procedures.
