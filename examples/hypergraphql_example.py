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


def example_hyperql_queries():
    """Example: HyperQL query language (Phase 4)"""
    print("=" * 60)
    print("Example 11: HyperQL Query Language (Phase 4)")
    print("=" * 60)
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-phase4-model.bin",
            model_type="hypergraphql",
            enable_hyperql=True
        )
        
        # Simple HyperQL query
        query1 = """
        SELECT ?person ?age
        WHERE {
            ?person :isA :Researcher .
            ?person :hasAge ?age .
            FILTER(?age > 30)
        }
        ORDER BY DESC(?age)
        LIMIT 10
        """
        
        print("Query 1: Find researchers over 30")
        results1 = llm.query_hyperql(query1)
        print(f"Found {len(results1)} results:")
        for r in results1[:3]:
            print(f"  - {r['person']}: {r['age']} years old")
        
        # Complex query with aggregation
        query2 = """
        SELECT ?paper (COUNT(?author) as ?num_authors)
        WHERE {
            HYPEREDGE ?he {
                :publication ?paper .
                :author ?author .
                :year ?year .
                :citations ?citations
            }
            FILTER(?year >= 2020)
        }
        GROUP BY ?paper
        HAVING COUNT(?author) >= 3
        ORDER BY DESC(?citations)
        LIMIT 5
        """
        
        print("\nQuery 2: Top collaborative papers")
        results2 = llm.query_hyperql(query2)
        for r in results2:
            print(f"  - {r['paper']}: {r['num_authors']} authors")
        
    except Exception as e:
        print(f"Note: Requires Phase 4 model with HyperQL support")
        print(f"Error: {e}")
    
    print()


def example_gpu_acceleration():
    """Example: GPU-accelerated inference (Phase 4)"""
    print("=" * 60)
    print("Example 12: GPU Acceleration (Phase 4)")
    print("=" * 60)
    
    try:
        # Enable CUDA acceleration
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            model_type="hypergraphql",
            device="cuda",
            cuda_config={
                "device_id": 0,
                "enable_graph_ops": True,
                "enable_attention": True,
                "use_flash_attention": True,
                "precision": "fp16"
            }
        )
        
        # Batch inference on GPU
        queries = [
            "What is deep learning?",
            "Explain neural networks",
            "How does backpropagation work?",
        ]
        
        print("Running batch inference on GPU...")
        responses = llm.batch_inference(queries, max_new_tokens=150)
        
        for query, response in zip(queries, responses):
            print(f"\nQ: {query}")
            print(f"A: {response[:100]}...")
        
        # Check GPU stats
        stats = llm.get_gpu_stats()
        print(f"\nGPU Performance:")
        print(f"  Utilization: {stats['utilization']}%")
        print(f"  Memory Used: {stats['memory_used_gb']:.2f} GB")
        print(f"  Throughput: {stats['tokens_per_second']:.1f} tok/s")
        
    except Exception as e:
        print(f"Note: Requires Phase 4 model with GPU support and CUDA")
        print(f"Error: {e}")
    
    print()


def example_large_scale_graph():
    """Example: Large-scale graph processing (Phase 4)"""
    print("=" * 60)
    print("Example 13: Large-Scale Graph Processing (Phase 4)")
    print("=" * 60)
    
    try:
        # Configure for large graph
        llm = AutoModelForCausalLM.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            model_type="hypergraphql",
            graph_config={
                "size": "xlarge",
                "partitioning": {
                    "strategy": "metis",
                    "num_partitions": 16
                },
                "compression": {
                    "nodes": "quantized",
                    "edges": "pruned",
                    "threshold": 0.01
                },
                "indexing": {
                    "type": "hnsw",
                    "ef_construction": 200
                },
                "processing_mode": "streaming",
                "chunk_size": 50000
            }
        )
        
        print("Loading massive knowledge graph...")
        llm.load_graph("/path/to/massive-graph.bin")
        
        # Query efficiently
        query = """
        SELECT ?entity ?type
        WHERE {
            ?entity :hasType ?type .
        }
        ORDER BY DESC(COUNT(?connections))
        LIMIT 100
        """
        
        print("Executing query on large graph...")
        results = llm.query_hyperql(query, timeout=60)
        print(f"Found {len(results)} highly connected entities")
        
        # Statistics
        stats = llm.get_graph_stats()
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {stats['num_nodes']:,}")
        print(f"  Edges: {stats['num_edges']:,}")
        print(f"  Partitions: {stats['num_partitions']}")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")
        
    except Exception as e:
        print(f"Note: Requires Phase 4 model with large-scale optimization")
        print(f"Error: {e}")
    
    print()


def example_distributed_inference():
    """Example: Distributed inference (Phase 4)"""
    print("=" * 60)
    print("Example 14: Distributed Inference (Phase 4)")
    print("=" * 60)
    
    try:
        from ctransformers import DistributedHypergraphQL
        
        # Set up distributed cluster
        llm = DistributedHypergraphQL.from_pretrained(
            "/path/to/hypergraphql-model.bin",
            cluster_config={
                "coordinator": "master-node:8000",
                "workers": [
                    {"address": "worker-1:8001", "gpus": [0, 1]},
                    {"address": "worker-2:8002", "gpus": [0, 1]},
                    {"address": "worker-3:8003", "gpus": [0, 1]},
                ],
                "distribution_strategy": "model_parallel",
                "load_balancing": "dynamic"
            }
        )
        
        print(f"Distributed cluster initialized")
        print(f"Total GPUs: {llm.get_total_gpus()}")
        
        # Complex distributed query
        query = """
        Analyze the evolution of AI research from 1950 to 2025.
        Include major areas, key researchers, and citation patterns.
        """
        
        print("\nExecuting distributed query...")
        response = llm(
            query,
            max_new_tokens=500,
            distributed=True,
            aggregation_strategy="ensemble"
        )
        
        print(f"Response: {response[:200]}...")
        
        # Cluster statistics
        stats = llm.get_cluster_stats()
        print(f"\nCluster Performance:")
        print(f"  Throughput: {stats['total_throughput']:.1f} tok/s")
        print(f"  Avg Latency: {stats['avg_latency_ms']:.1f} ms")
        print(f"  Load Balance: {stats['load_balance_ratio']:.2f}")
        
    except Exception as e:
        print(f"Note: Requires Phase 4 model with distributed support")
        print(f"Error: {e}")
    
    print()


def example_production_server():
    """Example: Production server deployment (Phase 4)"""
    print("=" * 60)
    print("Example 15: Production Server (Phase 4)")
    print("=" * 60)
    
    try:
        from ctransformers.serving import HypergraphQLServer
        from ctransformers.monitoring import MetricsCollector
        
        # Set up metrics
        metrics = MetricsCollector(
            backend="prometheus",
            config={"port": 9090}
        )
        
        # Create production server
        server = HypergraphQLServer(
            model_path="/path/to/hypergraphql-model.bin",
            config={
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 8,
                "max_batch_size": 32,
                "enable_cors": True
            }
        )
        
        # Configure caching
        server.configure_cache({
            "levels": [
                {"type": "memory", "size": "4GB", "ttl": 300},
                {"type": "redis", "host": "redis:6379", "size": "20GB"}
            ]
        })
        
        # Health checks
        server.configure_health_checks({
            "endpoint": "/health",
            "checks": ["model_loaded", "gpu_available"]
        })
        
        # Attach metrics
        server.attach_metrics(metrics)
        
        print("Production server configured:")
        print("  - REST API: http://localhost:8080/v1/query")
        print("  - HyperQL: http://localhost:8080/v1/hyperql")
        print("  - Health: http://localhost:8080/health")
        print("  - Metrics: http://localhost:9090/metrics")
        
        # Start server (commented out for example)
        # server.start()
        
    except Exception as e:
        print(f"Note: Requires Phase 4 model with production server")
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
    
    # Phase 4 examples
    example_hyperql_queries()
    example_gpu_acceleration()
    example_large_scale_graph()
    example_distributed_inference()
    example_production_server()
    
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
    print("\nPhase 3:")
    print("✓ OpenCog AtomSpace integration")
    print("✓ Temporal hypergraph evolution")
    print("✓ Dynamic graph structure modification")
    print("✓ Hierarchical relation types")
    print("✓ Context-based relation inference")
    print("✓ Bidirectional AtomSpace synchronization")
    print("✓ Time-aware embeddings and attention")
    print("\nPhase 4 (IN PROGRESS):")
    print("🚧 HyperQL query language (SPARQL-like)")
    print("🚧 GPU acceleration with CUDA/Metal (3-5x speedup)")
    print("🚧 Large-scale graph optimization (billion-edge graphs)")
    print("🚧 Distributed inference (multi-node clusters)")
    print("🚧 Production deployment tools and monitoring")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
