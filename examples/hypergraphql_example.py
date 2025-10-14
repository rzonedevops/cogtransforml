#!/usr/bin/env python3
"""
Example usage of the OpenCog HypergraphQL Transformer Model

This example demonstrates how to use the HypergraphQL model for:
1. Knowledge graph querying
2. Relational reasoning
3. Graph-to-text generation
"""

from ctransformers import AutoModelForCausalLM


def example_knowledge_graph_query():
    """Example: Query a knowledge graph using HypergraphQL"""
    print("=" * 60)
    print("Example 1: Knowledge Graph Querying")
    print("=" * 60)
    
    # Note: This requires a trained HypergraphQL model file
    # For demonstration purposes, we show the intended usage
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            model_type="hypergraphql"
        )
        
        # Query the knowledge graph
        query = "Query: What concepts are related to OpenCog?"
        response = llm(query, max_new_tokens=100)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Note: This requires a trained HypergraphQL model file")
        print(f"Error: {e}")
    
    print()


def example_multi_relational_query():
    """Example: Query with multiple relation types (Phase 2)"""
    print("=" * 60)
    print("Example 2: Multi-Relational Querying (Phase 2)")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            model_type="hypergraphql"
        )
        
        # Query using specific relation types
        query = """Query: Using 'is-a' and 'part-of' relations, 
        what is the relationship between neurons and the brain?"""
        response = llm(query, max_new_tokens=150)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
        # Query with causal relations
        query2 = "Query: Using 'causes' relations, what leads to learning?"
        response2 = llm(query2, max_new_tokens=100)
        print(f"\nQuery: {query2}")
        print(f"Response: {response2}")
        
    except Exception as e:
        print(f"Note: This requires a trained HypergraphQL model file")
        print(f"Error: {e}")
    
    print()


def example_relational_reasoning():
    """Example: Perform multi-hop reasoning over graph structures"""
    print("=" * 60)
    print("Example 2: Relational Reasoning")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            model_type="hypergraphql"
        )
        
        # Multi-hop reasoning query
        query = "If A relates to B, and B relates to C, what is the relationship between A and C?"
        response = llm(query, max_new_tokens=150)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Note: This requires a trained HypergraphQL model file")
        print(f"Error: {e}")
    
    print()


def example_embeddings():
    """Example: Get embeddings for graph concepts"""
    print("=" * 60)
    print("Example 3: Graph Concept Embeddings")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            model_type="hypergraphql"
        )
        
        # Get embeddings for a concept
        concept = "Artificial General Intelligence"
        embeddings = llm.embed(concept)
        print(f"Concept: {concept}")
        print(f"Embedding dimension: {len(embeddings)}")
        print(f"First 5 values: {embeddings[:5]}")
        
    except Exception as e:
        print(f"Note: This requires a trained HypergraphQL model file")
        print(f"Error: {e}")
    
    print()


def example_streaming():
    """Example: Stream responses for interactive queries"""
    print("=" * 60)
    print("Example 4: Streaming Generation")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            model_type="hypergraphql"
        )
        
        query = "Explain the hypergraph structure of knowledge representation:"
        print(f"Query: {query}")
        print("Response: ", end="", flush=True)
        
        for text in llm(query, stream=True, max_new_tokens=100):
            print(text, end="", flush=True)
        print()
        
    except Exception as e:
        print(f"\nNote: This requires a trained HypergraphQL model file")
        print(f"Error: {e}")
    
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("OpenCog HypergraphQL Transformer Model Examples")
    print("=" * 60 + "\n")
    
    print("This example demonstrates the intended usage of HypergraphQL.")
    print("Note: You need a trained model file to run these examples.\n")
    
    example_knowledge_graph_query()
    example_multi_relational_query()
    example_relational_reasoning()
    example_embeddings()
    example_streaming()
    
    print("=" * 60)
    print("Model Features:")
    print("=" * 60)
    print("Phase 1:")
    print("✓ Hypergraph-aware attention mechanism")
    print("✓ Graph convolution layers for message passing")
    print("✓ Node and hyperedge embeddings")
    print("✓ Standard transformer architecture")
    print("✓ Compatible with existing GGML framework")
    print("\nPhase 2 (NEW):")
    print("✓ Multi-relational hyperedge types")
    print("✓ Relation type embeddings")
    print("✓ Relation-aware attention mechanism")
    print("✓ Relation-aware graph convolution")
    print("✓ Support for typed relationships (is-a, part-of, causes, etc.)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
