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


def example_atomspace_integration():
    """Example: OpenCog AtomSpace integration (Phase 3)"""
    print("=" * 60)
    print("Example 5: AtomSpace Integration (Phase 3)")
    print("=" * 60)
    
    try:
        # Connect to AtomSpace
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-phase3-model.bin",
            model_type="hypergraphql",
            atomspace_uri="atomspace://localhost:5000"
        )
        
        # Query atoms directly
        print("Querying AtomSpace for animals...")
        result = llm.query_atoms(
            pattern="(InheritanceLink ?x (ConceptNode 'Animal'))",
            limit=10
        )
        print(f"Found {len(result)} matching atoms")
        
        # Enable bidirectional sync
        llm.enable_atomspace_sync(read=True, write=True)
        
        # Query with AtomSpace context
        query = "What are the properties of mammals?"
        response = llm(query, max_new_tokens=150)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Note: Requires trained Phase 3 model and AtomSpace connection")
        print(f"Error: {e}")
    
    print()


def example_temporal_reasoning():
    """Example: Temporal hypergraph evolution (Phase 3)"""
    print("=" * 60)
    print("Example 6: Temporal Reasoning (Phase 3)")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-phase3-model.bin",
            model_type="hypergraphql",
            enable_temporal=True
        )
        
        # Query historical relationships
        query_past = "What was the relationship between USSR and Russia in 1980?"
        response_past = llm(
            query_past,
            temporal_context={"year": 1980},
            max_new_tokens=100
        )
        print(f"Historical Query: {query_past}")
        print(f"Response: {response_past}")
        
        # Query current relationships
        query_now = "What is the relationship between USSR and Russia now?"
        response_now = llm(
            query_now,
            temporal_context={"year": 2025},
            max_new_tokens=100
        )
        print(f"\nCurrent Query: {query_now}")
        print(f"Response: {response_now}")
        
        # Track evolution over time
        print("\nTracking evolution...")
        evolution = llm.track_evolution(
            subject="USSR",
            object="Russia",
            time_range=("1980-01-01", "2025-01-01"),
            granularity="decade"
        )
        
        for period in evolution:
            print(f"  {period['time']}: {period['relation']} (conf: {period['confidence']:.2f})")
        
    except Exception as e:
        print(f"Note: Requires trained Phase 3 model with temporal features")
        print(f"Error: {e}")
    
    print()


def example_dynamic_graph():
    """Example: Dynamic graph modification (Phase 3)"""
    print("=" * 60)
    print("Example 7: Dynamic Graph Modification (Phase 3)")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-phase3-model.bin",
            model_type="hypergraphql",
            enable_dynamic=True
        )
        
        # Initial query
        print("Initial query about mammals...")
        response = llm("What unusual mammals exist?", max_new_tokens=100)
        print(f"Response: {response}")
        
        # Add new knowledge dynamically
        print("\nAdding new knowledge about platypus...")
        llm.add_node(
            node_id="Platypus",
            node_type="ConceptNode",
            attributes={"category": "animal", "novelty": 0.9}
        )
        llm.add_edge("Platypus", "Mammal", "is-a", confidence=0.95)
        llm.add_edge("Platypus", "EggLaying", "has-property", confidence=0.99)
        
        # Query with updated knowledge
        print("Query after adding knowledge...")
        response2 = llm("What unusual mammals lay eggs?", max_new_tokens=100)
        print(f"Response: {response2}")
        
    except Exception as e:
        print(f"Note: Requires trained Phase 3 model with dynamic features")
        print(f"Error: {e}")
    
    print()


def example_hierarchical_relations():
    """Example: Hierarchical relation types (Phase 3)"""
    print("=" * 60)
    print("Example 8: Hierarchical Relations (Phase 3)")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-phase3-model.bin",
            model_type="hypergraphql"
        )
        
        # Define custom relation hierarchy
        llm.set_relation_hierarchy({
            "Physical": {
                "part-of": {"weight": 1.0},
                "attached-to": {"weight": 0.8},
                "connected-to": {"weight": 0.6}
            },
            "Conceptual": {
                "is-a": {"weight": 1.0},
                "similar-to": {"weight": 0.7},
                "related-to": {"weight": 0.5}
            }
        })
        
        # Query with hierarchy awareness
        query = "Find all physical relationships between engine and car"
        response = llm(
            query,
            relation_category="Physical",
            include_descendants=True,
            max_new_tokens=150
        )
        print(f"Query: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Note: Requires trained Phase 3 model")
        print(f"Error: {e}")
    
    print()


def example_relation_inference():
    """Example: Context-based relation inference (Phase 3)"""
    print("=" * 60)
    print("Example 9: Relation Inference (Phase 3)")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-phase3-model.bin",
            model_type="hypergraphql"
        )
        
        # Automatic relation inference
        print("Inferring relation from context...")
        result = llm.infer_relation(
            source="heart",
            target="body",
            context="The heart is an organ in the body that pumps blood",
            return_confidence=True,
            return_alternatives=True
        )
        
        print(f"Source: heart")
        print(f"Target: body")
        print(f"Context: The heart is an organ in the body that pumps blood")
        print(f"\nPrimary relation: {result['relation']} (confidence: {result['confidence']:.2f})")
        print("Alternative interpretations:")
        for alt in result['alternatives']:
            print(f"  - {alt['relation']}: {alt['confidence']:.2f}")
        
    except Exception as e:
        print(f"Note: Requires trained Phase 3 model")
        print(f"Error: {e}")
    
    print()


def example_combined_phase3():
    """Example: Combined Phase 3 features"""
    print("=" * 60)
    print("Example 10: Combined Phase 3 Features")
    print("=" * 60)
    
    try:
        # Initialize with all Phase 3 features
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-phase3-model.bin",
            model_type="hypergraphql",
            phase3_config={
                "atomspace_uri": "atomspace://localhost:5000",
                "enable_temporal": True,
                "enable_dynamic": True,
                "temporal_steps": 1000,
                "hierarchy_depth": 4
            }
        )
        
        # Enable AtomSpace sync
        llm.enable_atomspace_sync(read=True, write=True)
        
        # Complex query using all Phase 3 features
        query = """
        Using knowledge from AtomSpace:
        1. What was the relationship between neurons and the brain in 1950?
        2. What is the current understanding (2025)?
        3. How has this evolved?
        
        Consider hierarchical semantic relationships and infer missing relations.
        """
        
        print(f"Query: {query}")
        response = llm(
            query,
            temporal_context={"compare_years": [1950, 2025]},
            hierarchy_depth=3,
            infer_relations=True,
            update_atomspace=True,
            max_new_tokens=300
        )
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Note: Requires trained Phase 3 model with all features")
        print(f"Error: {e}")
    
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("OpenCog HypergraphQL Transformer Model Examples")
    print("=" * 60 + "\n")
    
    print("This example demonstrates the intended usage of HypergraphQL.")
    print("Note: You need a trained model file to run these examples.\n")
    
    # Phase 1 & 2 examples
    example_knowledge_graph_query()
    example_multi_relational_query()
    example_relational_reasoning()
    example_embeddings()
    example_streaming()
    
    # Phase 3 examples
    example_atomspace_integration()
    example_temporal_reasoning()
    example_dynamic_graph()
    example_hierarchical_relations()
    example_relation_inference()
    example_combined_phase3()
    
    print("=" * 60)
    print("Model Features:")
    print("=" * 60)
    print("Phase 1:")
    print("✓ Hypergraph-aware attention mechanism")
    print("✓ Graph convolution layers for message passing")
    print("✓ Node and hyperedge embeddings")
    print("✓ Standard transformer architecture")
    print("✓ Compatible with existing GGML framework")
    print("\nPhase 2:")
    print("✓ Multi-relational hyperedge types")
    print("✓ Relation type embeddings")
    print("✓ Relation-aware attention mechanism")
    print("✓ Relation-aware graph convolution")
    print("✓ Support for typed relationships (is-a, part-of, causes, etc.)")
    print("\nPhase 3 (NEW):")
    print("✓ OpenCog AtomSpace integration")
    print("✓ Temporal hypergraph evolution")
    print("✓ Dynamic graph structure modification")
    print("✓ Hierarchical relation types")
    print("✓ Context-based relation inference")
    print("✓ Bidirectional AtomSpace synchronization")
    print("✓ Time-aware embeddings and attention")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
