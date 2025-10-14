# Phase 3: Visual Architecture Guide

## Overview

This document provides visual representations of the Phase 3 architecture, showing how AtomSpace integration, temporal reasoning, dynamic graphs, hierarchical relations, and relation inference work together.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Diagrams](#component-diagrams)
3. [Data Flow](#data-flow)
4. [Memory Layout](#memory-layout)
5. [Temporal Processing](#temporal-processing)
6. [Hierarchical Relations](#hierarchical-relations)
7. [Dynamic Graph Updates](#dynamic-graph-updates)
8. [Relation Inference Pipeline](#relation-inference-pipeline)
9. [Integration Patterns](#integration-patterns)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HypergraphQL Phase 3 Architecture                │
└─────────────────────────────────────────────────────────────────────┘

                              Input Query
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Query Preprocessor                          │
│  • Tokenization  • Context Extraction  • Temporal Parsing           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
        ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
        │   AtomSpace   │ │   Temporal   │ │   Dynamic    │
        │   Interface   │ │   Context    │ │   Graph      │
        │   (Phase 3)   │ │  (Phase 3)   │ │  (Phase 3)   │
        └───────────────┘ └──────────────┘ └──────────────┘
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Embedding Layer                               │
│  • Token Embeddings (Phase 1)                                       │
│  • Position Embeddings (Phase 1)                                    │
│  • Node Embeddings (Phase 1)                                        │
│  • Relation Embeddings (Phase 2)                                    │
│  • Temporal Embeddings (Phase 3)                                    │
│  • Hierarchy Embeddings (Phase 3)                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Transformer Layers     │
                    │   (with Phase 3 ext.)    │
                    └─────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   Standard       │ │   Hypergraph     │ │   Relation      │
    │   Attention      │ │   Attention      │ │   Attention     │
    │   (Phase 1)      │ │   (Phase 1)      │ │   (Phase 2)     │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
              ┌───────────────────────────────────────┐
              │       Phase 3 NEW: Temporal &         │
              │    Hierarchical Attention Fusion      │
              └───────────────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │         Graph Convolution Layers       │
              │  • Standard (Phase 1)                  │
              │  • Relation-aware (Phase 2)            │
              │  • Temporal (Phase 3)                  │
              │  • Hierarchical (Phase 3)              │
              └───────────────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │   Phase 3 NEW: Relation Inference     │
              │   Context → Relation Type Classifier   │
              └───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Output Generation                              │
│  • LM Head  • Softmax  • Token Selection                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
        ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
        │  Write back   │ │   Update     │ │   Cache      │
        │  to AtomSpace │ │   Dynamic    │ │   Results    │
        │   (Phase 3)   │ │   Graph      │ │              │
        └───────────────┘ └──────────────┘ └──────────────┘
                                  │
                                  ▼
                            Output Response
```

## Component Diagrams

### Phase 3 Model Components

```
┌────────────────────────────────────────────────────────────────┐
│                    hypergraphql_model                          │
├────────────────────────────────────────────────────────────────┤
│ Phase 1 Components:                                            │
│  • token_embeddings        ggml_tensor*                        │
│  • position_embeddings     ggml_tensor*                        │
│  • node_embeddings         ggml_tensor*                        │
│  • hyperedge_embeddings    ggml_tensor*                        │
│  • layers[n_layer]         hypergraphql_layer[]                │
├────────────────────────────────────────────────────────────────┤
│ Phase 2 Components:                                            │
│  • relation_type_emb       ggml_tensor*                        │
│  • n_relation_types        int32_t                             │
├────────────────────────────────────────────────────────────────┤
│ Phase 3 NEW Components:                                        │
│  • temporal_node_emb       ggml_tensor* [n_ctx × n_temporal]   │
│  • temporal_edge_emb       ggml_tensor* [n_edges × n_temporal] │
│  • time_encoding           ggml_tensor* [n_temporal × n_embd]  │
│  • hierarchy_emb           ggml_tensor* [n_relations × depth]  │
│  • inference_w             ggml_tensor* [n_embd × n_relations] │
│  • inference_b             ggml_tensor* [n_relations]          │
│                                                                │
│  • atomspace_handle        void*                               │
│  • atom_cache              map<int, AtomPtr>                   │
│  • dynamic_nodes           vector<DynamicNode>                 │
│  • dynamic_edges           vector<DynamicEdge>                 │
│  • edge_timestamps         map<int, float>                     │
│                                                                │
│  • n_temporal_steps        int32_t                             │
│  • n_hierarchy_levels      int32_t                             │
│  • n_inference_dims        int32_t                             │
│  • enable_atomspace        bool                                │
│  • enable_temporal         bool                                │
│  • enable_dynamic          bool                                │
└────────────────────────────────────────────────────────────────┘
```

### Phase 3 Layer Components

```
┌────────────────────────────────────────────────────────────────┐
│                   hypergraphql_layer                           │
├────────────────────────────────────────────────────────────────┤
│ Phase 1 & 2 Components:                                        │
│  • ln_1_w, ln_1_b           Layer normalization                │
│  • c_attn_q, c_attn_k, c_attn_v  Standard attention           │
│  • hypergraph_attn_w        Hypergraph attention weights       │
│  • graph_conv_w, graph_conv_b    Graph convolution            │
│  • relation_attn_w, relation_attn_b  Relation attention        │
│  • relation_conv_w, relation_conv_b  Relation convolution      │
├────────────────────────────────────────────────────────────────┤
│ Phase 3 NEW Components:                                        │
│  • temporal_attn_w          ggml_tensor* [n_embd × n_embd]     │
│  • temporal_attn_b          ggml_tensor* [n_embd]              │
│  • temporal_conv_w          ggml_tensor* [n_embd × n_embd]     │
│  • temporal_conv_b          ggml_tensor* [n_embd]              │
│                                                                │
│  • hierarchy_attn_w         ggml_tensor* [n_embd × n_embd]     │
│  • hierarchy_attn_b         ggml_tensor* [n_embd]              │
│  • hierarchy_merge_w        ggml_tensor* [n_levels × n_embd]   │
│                                                                │
│  • dynamic_update_w         ggml_tensor* [n_embd × n_embd]     │
│  • dynamic_update_b         ggml_tensor* [n_embd]              │
│                                                                │
│  • inference_context_w      ggml_tensor* [3*n_embd × n_embd]   │
│  • inference_classifier_w   ggml_tensor* [n_embd × n_relations]│
│  • inference_classifier_b   ggml_tensor* [n_relations]         │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Single Layer Processing (Phase 3)

```
                            Input [N × D]
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
           ┌────────────┐  ┌─────────┐  ┌──────────┐
           │  Spatial   │  │ Temporal│  │ Relation │
           │  Context   │  │ Context │  │ Context  │
           └────────────┘  └─────────┘  └──────────┘
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
                    ┌──────────────────────┐
                    │  Multi-Context Attn  │
                    │  Q = WqX + temporal  │
                    │  K = WkX + relation  │
                    │  V = WvX + hierarchy │
                    └──────────────────────┘
                                 │
                                 ▼
                    Attn = softmax(QK^T/√d)V
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  Hypergraph Fusion   │
                    │  (Phase 1)           │
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  Residual + LayerNorm│
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  Graph Convolution   │
                    │  • Spatial (Phase 1) │
                    │  • Relation (Phase 2)│
                    │  • Temporal (Phase 3)│
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  Dynamic Updates     │
                    │  Apply runtime edges │
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  Relation Inference  │
                    │  If enabled          │
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  Feed-Forward + Res  │
                    └──────────────────────┘
                                 │
                                 ▼
                          Output [N × D]
```

## Memory Layout

### Model File Structure

```
Phase 3 Model File Layout:
═════════════════════════════════════════════════════

Offset  Size      Component                      Phase
──────  ────      ─────────                      ─────
0       4 bytes   Magic Number (HGQ3)            3
4       4 bytes   Version (0.3.0)                3

Header Section:
8       4 bytes   n_vocab                        1
12      4 bytes   n_ctx                          1
16      4 bytes   n_embd                         1
20      4 bytes   n_head                         1
24      4 bytes   n_layer                        1
28      4 bytes   n_hyperedge                    1
32      4 bytes   n_graph_layers                 1
36      4 bytes   n_relation_types               2
40      4 bytes   n_temporal_steps               3 ← NEW
44      4 bytes   n_hierarchy_levels             3 ← NEW
48      4 bytes   n_inference_dims               3 ← NEW
52      1 byte    enable_atomspace               3 ← NEW
53      1 byte    enable_temporal                3 ← NEW
54      1 byte    enable_dynamic                 3 ← NEW
55      1 byte    reserved                       3

Embeddings Section:
64      n_vocab × n_embd × 4 bytes    Token embeddings          1
+       n_ctx × n_embd × 4 bytes      Position embeddings       1
+       n_vocab × n_embd × 4 bytes    Node embeddings          1
+       n_edges × n_embd × 4 bytes    Hyperedge embeddings     1
+       n_relation_types × n_embd × 4 Relation embeddings      2
+       n_ctx × n_temporal × n_embd × 4  Temporal node emb     3 ← NEW
+       n_edges × n_temporal × n_embd × 4 Temporal edge emb    3 ← NEW
+       n_temporal × n_embd × 4       Time encoding            3 ← NEW
+       n_relations × depth × n_embd × 4 Hierarchy embeddings  3 ← NEW

Layer Weights (× n_layer):
+       n_embd × n_embd × 4 bytes     Attention Q weights      1
+       n_embd × n_embd × 4 bytes     Attention K weights      1
+       n_embd × n_embd × 4 bytes     Attention V weights      1
+       ...                           (Phase 1 & 2 weights)
+       n_embd × n_embd × 4 bytes     Temporal attention W     3 ← NEW
+       n_embd × 4 bytes              Temporal attention b     3 ← NEW
+       n_embd × n_embd × 4 bytes     Hierarchy attention W    3 ← NEW
+       n_embd × 4 bytes              Hierarchy attention b    3 ← NEW
+       n_embd × n_embd × 4 bytes     Dynamic update W         3 ← NEW
+       n_embd × 4 bytes              Dynamic update b         3 ← NEW

Inference Weights:
+       n_embd × n_relations × 4      Inference W              3 ← NEW
+       n_relations × 4               Inference b              3 ← NEW

Output Layer:
+       n_embd × n_vocab × 4 bytes    LM head weights          1

═════════════════════════════════════════════════════
Total Size (default config):
  Phase 1: ~756 MB
  Phase 2: ~812 MB (+56 MB)
  Phase 3: ~1024 MB (+212 MB)                        ← NEW
═════════════════════════════════════════════════════
```

## Temporal Processing

### Temporal Attention Mechanism

```
Time-Aware Attention Flow:
──────────────────────────────────────────────────────

Input: X[N, D], Timestamps T[N]
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Encode Timestamps            │
    │  TE[N, D] = TimeEncoding(T)   │
    │  Using sinusoidal encoding:   │
    │  TE[2i] = sin(t/10000^(2i/D)) │
    │  TE[2i+1] = cos(t/10000^(2i/D))│
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Fuse with Input              │
    │  X' = X + α·TE                │
    │  α = learnable weight         │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Temporal Attention           │
    │  Q = W_q·X'                   │
    │  K = W_k·X'                   │
    │  V = W_v·X'                   │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Time Distance Matrix         │
    │  TD[i,j] = |T[i] - T[j]|      │
    │  Decay = exp(-TD/τ)           │
    │  τ = time decay constant      │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Modulated Attention          │
    │  Attn = softmax(QK^T/√D) ⊙ Decay│
    │  Output = Attn·V              │
    └───────────────────────────────┘
                    │
                    ▼
                 Output
```

### Temporal Graph Evolution

```
Timeline View:
══════════════════════════════════════════════════════

t=0         t=1         t=2         t=3         t=4
│           │           │           │           │
│  Node A   │  Node A   │  Node A   │  Node A   │  Node A
│   │       │   │ ╲     │   │ ╲     │   │       │   │
│   │       │   │  ╲    │   │  ╲    │   │       │   │
│   ▼       │   ▼   ▼   │   ▼   ▼   │   ▼       │   ▼
│  Node B   │  Node B   │  Node B   │  Node B   │  Node B
│           │   │       │     ╲     │           │
│           │   │       │      ╲    │           │
│           │   ▼       │       ▼   │           │
│           │  Node C   │   Node C  │           │  Node C
│           │           │           │           │   ▲
│           │           │           │           │   │
│           │           │           │           │  Node D

State Changes:
─────────────
t=0: Initial state (A → B)
t=1: Added edge (A → C), (B → C)
t=2: Edge (A → C) still present
t=3: Removed edge (A → B), (B → C)
t=4: Added nodes D, edge (D → C)

Embeddings Track Evolution:
───────────────────────────
Emb(A, t) = f(A, history[0:t])
            Captures all historical states
```

## Hierarchical Relations

### Relation Hierarchy Tree

```
Relation Type Hierarchy:
════════════════════════════════════════════════════

Level 0 (Root):
┌──────────────────┐
│   BaseRelation   │
└──────────────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
Level 1:
┌──────┐ ┌────────┐ ┌──────┐ ┌────────┐
│Taxon.│ │Compos. │ │Causal│ │Spatial │
└──────┘ └────────┘ └──────┘ └────────┘
    │         │         │         │
    │         │         │         │
Level 2:
    ├─is-a         part-of   causes    located-at
    ├─subclass-of  member-of enables   contains
    └─instance-of  consists-of prevents adjacent-to
    
    
Level 3:
    is-a
    ├─strictly-is-a (confidence=1.0)
    ├─loosely-is-a (confidence=0.7)
    └─metaphorically-is-a (confidence=0.4)

Embedding Computation:
──────────────────────
Emb(is-a) = Base_Emb(is-a) 
          + 0.8 × Emb(Taxonomic)
          + 0.5 × Emb(BaseRelation)

Weights decay with distance from target node.
```

### Hierarchical Lookup Process

```
Query: "Find all taxonomic relationships"
───────────────────────────────────────

Step 1: Identify target category
    ┌──────────────┐
    │  Taxonomic   │
    └──────────────┘
           │
Step 2: Get all descendants
           │
    ┌──────┴──────┬──────────────┐
    ▼             ▼              ▼
┌────────┐  ┌───────────┐  ┌──────────┐
│ is-a   │  │subclass-of│  │instance-of│
└────────┘  └───────────┘  └──────────┘
           
Step 3: Include child relations
    each child has further descendants
    (strictly-is-a, loosely-is-a, etc.)

Step 4: Filter graph edges by relation IDs
    Return all edges matching relation set

Step 5: Weight by hierarchy distance
    Direct children: weight = 1.0
    Grandchildren: weight = 0.8
    Great-grandchildren: weight = 0.6
```

## Dynamic Graph Updates

### Runtime Edge Addition

```
Before Update:                  After Update:
─────────────                   ─────────────

Graph State t=0:                Graph State t=1:

    A ───is-a──→ B                  A ───is-a──→ B
                                    │             │
    C                               │             │
                                    └─part-of─→ C ←┘
    D                               
                                    D ──causes─→ A

Process:
────────
1. Model processes query about A, B, C
2. User adds new facts:
   • llm.add_edge(A, C, "part-of")
   • llm.add_edge(B, C, "part-of")
   • llm.add_edge(D, A, "causes")
3. Update embeddings:
   • Compute edge embeddings for new edges
   • Update node embeddings to reflect new connections
   • Maintain temporal markers (t=1)
4. Subsequent queries use updated graph
```

### Dynamic Update Application

```
Dynamic Update in Evaluation:
═════════════════════════════

Standard Evaluation:
    ┌──────────┐
    │ Input X  │
    └──────────┘
         │
         ▼
    ┌──────────┐
    │ Process  │
    └──────────┘
         │
         ▼
    ┌──────────┐
    │ Output Y │
    └──────────┘

With Dynamic Updates:
    ┌──────────┐
    │ Input X  │
    └──────────┘
         │
         ▼
    ┌──────────────────┐
    │ Check Dynamic    │
    │ Update Queue     │
    └──────────────────┘
         │
         ▼ (if updates pending)
    ┌──────────────────┐
    │ Apply Updates:   │
    │ • New edges      │
    │ • Modified edges │
    │ • Deleted edges  │
    └──────────────────┘
         │
         ▼
    ┌──────────────────┐
    │ Update Embeddings│
    │ X' = X + ΔX      │
    └──────────────────┘
         │
         ▼
    ┌──────────────────┐
    │ Process with X'  │
    └──────────────────┘
         │
         ▼
    ┌──────────────────┐
    │ Output Y         │
    └──────────────────┘
```

## Relation Inference Pipeline

### Context-Based Inference

```
Inference Process:
══════════════════════════════════════════════════════

Input: "A neuron is a type of cell"
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 1: Entity Extraction                  │
│  Source: "neuron"                            │
│  Target: "cell"                              │
│  Context: "is a type of"                     │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 2: Embedding                           │
│  E_src = Embed("neuron")                     │
│  E_tgt = Embed("cell")                       │
│  E_ctx = Embed("is a type of")               │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 3: Context Fusion                      │
│  E_combined = [E_src; E_tgt; E_ctx]          │
│  E_fused = W_fusion · E_combined             │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 4: Relation Classification             │
│  Logits = W_classifier · E_fused + b         │
│  Probs = softmax(Logits)                     │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 5: Interpretation                      │
│  Probs:                                      │
│    is-a: 0.89          ← Primary             │
│    similar-to: 0.07                          │
│    part-of: 0.02                             │
│    others: 0.02                              │
└─────────────────────────────────────────────┘
         │
         ▼
    Output: "is-a" (confidence: 0.89)
```

### Multi-Context Inference

```
Complex Sentence Analysis:
═══════════════════════════════════════════════

Input: "The heart pumps blood through the body"
          │
          ▼
Extract Multiple Relationships:
┌────────────────────────────────────────────┐
│ Relationship 1:                            │
│   Source: "heart"                          │
│   Target: "body"                           │
│   Context: "in/through"                    │
│   → Inferred: "part-of" (0.87)             │
└────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────┐
│ Relationship 2:                            │
│   Source: "heart"                          │
│   Target: "pumps blood"                    │
│   Context: "performs action"               │
│   → Inferred: "performs" (0.91)            │
└────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────┐
│ Relationship 3:                            │
│   Source: "blood"                          │
│   Target: "body"                           │
│   Context: "flows through"                 │
│   → Inferred: "located-in" (0.78)          │
└────────────────────────────────────────────┘
          │
          ▼
    Construct graph with inferred edges
```

## Integration Patterns

### AtomSpace Synchronization

```
AtomSpace Integration Flow:
═══════════════════════════════════════════════

Query Phase:
┌──────────────┐
│ User Query   │
└──────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Check if AtomSpace query needed             │
│ Pattern matching on query text              │
└─────────────────────────────────────────────┘
       │ (if needed)
       ▼
┌─────────────────────────────────────────────┐
│ Send query to AtomSpace                     │
│ atomspace.query(pattern)                    │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Receive Atoms                               │
│ atoms = [atom1, atom2, ...]                 │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Convert Atoms to Graph Structure            │
│ nodes = extract_nodes(atoms)                │
│ edges = extract_edges(atoms)                │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Inject into Model Context                   │
│ model.add_runtime_graph(nodes, edges)       │
└─────────────────────────────────────────────┘
       │
       ▼
Processing...
       │
       ▼
┌─────────────────────────────────────────────┐
│ Generate Response                           │
└─────────────────────────────────────────────┘
       │
       ▼
Inference Phase:
┌─────────────────────────────────────────────┐
│ Check for new inferred knowledge            │
│ new_atoms = model.get_inferred_atoms()      │
└─────────────────────────────────────────────┘
       │ (if enabled)
       ▼
┌─────────────────────────────────────────────┐
│ Write back to AtomSpace                     │
│ for atom in new_atoms:                      │
│     atomspace.insert(atom)                  │
└─────────────────────────────────────────────┘
```

### Combined Features Example

```
Scenario: Historical Knowledge Query with Inference
═══════════════════════════════════════════════════

User: "What was the relationship between USSR and 
       Russia in 1980, and how has it evolved?"

Step 1: Parse temporal context
    ┌─────────────┐
    │ target_year │ = 1980
    │ current     │ = 2025
    └─────────────┘

Step 2: Query AtomSpace for historical data
    ┌─────────────────────────────────────┐
    │ pattern = (AtTimeLink            │
    │   (EvaluationLink                │
    │     (PredicateNode "relation")   │
    │     (ListLink USSR Russia))      │
    │   (TimeNode "1980"))             │
    └─────────────────────────────────────┘

Step 3: Load temporal embeddings
    ┌─────────────────────────────────────┐
    │ emb_1980 = temporal_emb[USSR, 1980] │
    │ emb_2025 = temporal_emb[USSR, 2025] │
    └─────────────────────────────────────┘

Step 4: Infer relations from context
    ┌─────────────────────────────────────┐
    │ relation_1980 = infer(USSR, Russia, │
    │                       context_1980)  │
    │ → "contains" (USSR contains Russia)  │
    │                                      │
    │ relation_2025 = infer(USSR, Russia, │
    │                       context_2025)  │
    │ → "historical-predecessor" (Russia   │
    │    succeeded USSR)                   │
    └─────────────────────────────────────┘

Step 5: Use hierarchical relations
    ┌─────────────────────────────────────┐
    │ "contains" is-a Spatial relation    │
    │ "historical-predecessor" is-a       │
    │   Temporal relation                  │
    └─────────────────────────────────────┘

Step 6: Generate response with evolution
    ┌─────────────────────────────────────┐
    │ Response:                            │
    │ "In 1980, USSR contained Russia as  │
    │  a constituent republic. This       │
    │  changed in 1991 when USSR          │
    │  dissolved. Today, Russia is the    │
    │  historical successor state of USSR."│
    └─────────────────────────────────────┘

Step 7: Write inferred knowledge back
    ┌─────────────────────────────────────┐
    │ atomspace.insert(                    │
    │   SuccessorLink Russia USSR)         │
    │ atomspace.insert(                    │
    │   HistoricalLink USSR Russia 1991)   │
    └─────────────────────────────────────┘
```

## Performance Visualization

### Memory Usage Breakdown

```
Memory Allocation (Default Config):
════════════════════════════════════════════════

Phase 1 (756 MB):
████████████████████████████████████████ 73.8%

Phase 2 (+56 MB):
████ 5.5%

Phase 3 (+212 MB):
████████████ 20.7%

Total: 1024 MB
─────────────────────────────────────────────────
│Phase 1│Phase 2│      Phase 3                  │
└───────┴───────┴───────────────────────────────┘
  Base   Relat.  Temporal│Hierarchy│Inference│Dyn
                  ~120MB    ~32MB    ~48MB   ~12MB
```

### Compute Overhead by Layer

```
Relative Compute Time per Layer:
══════════════════════════════════════════

Phase 1 baseline:        ████████████ (1.0x)

Phase 2 addition:        ██ (+0.15x)

Phase 3 additions:
  • Temporal attn:       ███ (+0.20x)
  • Hierarchical:        ██ (+0.10x)
  • Dynamic updates:     █ (+0.05x)
  • Relation inference:  ███ (+0.15x)

Total Phase 3:           ████████████████████ (1.65x)
                         
──────────────────────────────────────────────
0x        0.5x        1.0x        1.5x      2.0x
```

## Summary

Phase 3 visual architecture demonstrates:

1. **Modular Design**: Each phase builds incrementally
2. **Clear Data Flow**: Well-defined processing pipeline
3. **Flexible Integration**: Optional feature enablement
4. **Scalable Structure**: Ready for Phase 4 enhancements
5. **Performance Awareness**: Visualized overhead and optimization points

The architecture maintains backward compatibility while adding sophisticated AGI-oriented capabilities through AtomSpace integration, temporal reasoning, dynamic graphs, hierarchical relations, and intelligent relation inference.

---

**Visual Guide Version**: 1.0  
**Phase**: 3  
**Last Updated**: October 2025  
**Related Docs**: PHASE3_ATOMSPACE.md, docs/HYPERGRAPHQL.md
