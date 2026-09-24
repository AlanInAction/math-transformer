# math-transformer

Exploring Transformer from the perspective of **real analysis, linear algebra, vectors, matrices, and graph propagation** — implemented from scratch in pure Python.

[中文 README](README.zh-CN.md)

---

## Overview

`math-transformer` is a learning-oriented implementation of Transformer built from first principles.

The goal is not simply to reproduce a Transformer implementation, but to understand the mathematical structures that gradually lead from a corpus to a Transformer:

```text
Corpus
  ↓
Character Sequence
  ↓
Byte Pair Encoding
  ↓
Symbol Sequence
  ↓
Co-occurrence Graph
  ↓
Co-occurrence Matrix
  ↓
Embedding
  ↓
Token State
  ↓
Attention
  ↓
Transformer Block
```

The project deliberately starts with **pure Python**, avoiding deep-learning frameworks in the early stages. This keeps the underlying mathematical operations visible and makes it possible to connect each implementation step with the corresponding mathematical structure.

---

## Core Idea

The project is built around three complementary perspectives:

```text
                 ┌──────────────┐
                 │ State Machine│
                 └──────┬───────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Vector / Matrix  │
              │     Algebra      │
              └────────┬─────────┘
                       │
                       ▼
                ┌────────────┐
                │    Graph   │
                │ Propagation│
                └────────────┘
```

These are not treated as independent topics.

A Transformer can be viewed as a **state transition system** whose states are represented by vectors and matrices, while Attention dynamically constructs a graph over the sequence and propagates information along that graph.

This gives the following conceptual transition:

$$
\boxed{
\text{Static Graph}
\rightarrow
\text{Token Representation}
\rightarrow
\text{Dynamic Graph}
\rightarrow
\text{State Propagation}
}
$$

---

## Mathematical Perspective

A central idea in this project is to view a vector as a finite-dimensional function.

For an index set \(I\),

$$
V : I \rightarrow \mathbb{R}
$$

A vector is therefore an indexed collection of scalar values.

For example, if

$$
I=\{0,1,2\},
$$

then

$$
V:I\rightarrow\mathbb{R}
$$

can be represented as

$$
V=(v_0,v_1,v_2).
$$

A matrix naturally extends this idea to two index sets:

$$
M:I\times J\rightarrow\mathbb{R}.
$$

This perspective provides a unified way to understand vectors, matrices, and higher-dimensional tensors as functions over product index sets.

### Matrix multiplication

For

$$
A:I\times J\rightarrow\mathbb{R}
$$

and

$$
B:J\times K\rightarrow\mathbb{R},
$$

their product is

$$
AB:I\times K\rightarrow\mathbb{R},
$$

with

$$
(AB)(i,k)
=
\sum_{j\in J}A(i,j)B(j,k).
$$

In other words:

> Fix the output indices and sum over the shared index.

This interpretation becomes particularly useful when understanding Attention.

---

## From Static Graph to Dynamic Graph

The tokenizer and embedding stages construct a relatively static representation of relationships between symbols.

The co-occurrence structure can be viewed as a weighted graph:

$$
G_{\text{static}}=\langle V,E\rangle
$$

where:

* \(V\) represents tokens/symbols;
* \(E\) represents co-occurrence relationships;
* edge weights represent the strength of those relationships.

The embedding process then maps these symbolic relationships into vectors.

The Transformer introduces a different kind of graph.

Given a token state

$$
X:D\times N\rightarrow\mathbb{R},
$$

where:

* \(D\) is the embedding/state dimension;
* \(N\) is the sequence index set;

Attention constructs a graph whose edge weights depend on the current state.

Thus:

$$
A=A(X).
$$

The graph is no longer fixed.

It changes according to the current representation of the sequence.

---

## Attention

The current implementation follows the mathematical structure of scaled dot-product Attention.

Given

$$
X\in\mathbb{R}^{D\times N},
$$

three independently parameterized transformations produce:

$$
Q=W_QX,
$$

$$
K=W_KX,
$$

$$
V=W_VX.
$$

The interaction between sequence positions is then computed through:

$$
S=Q^TK.
$$

Since

$$
Q^T\in\mathbb{R}^{N\times D},
\qquad
K\in\mathbb{R}^{D\times N},
$$

we obtain:

$$
S\in\mathbb{R}^{N\times N}.
$$

This matrix can be interpreted as a dynamically constructed weighted graph over the sequence.

After scaling:

$$
\hat S=\frac{Q^TK}{\sqrt D},
$$

softmax converts each row into normalized positive weights:

$$
A=\operatorname{softmax}(\hat S).
$$

Finally, these weights propagate transformed token states:

$$
Y=VA.
$$

The complete operation can therefore be summarized as:

$$
\boxed{
X
\rightarrow
Q,K,V
\rightarrow
Q^TK
\rightarrow
\frac{Q^TK}{\sqrt D}
\rightarrow
\operatorname{softmax}
\rightarrow
VA
}
$$

From the graph perspective:

```text
Current State
     │
     ▼
 Q / K transformations
     │
     ▼
Dynamic Graph
     │
     ▼
Normalized Edge Weights
     │
     ▼
 V transformation
     │
     ▼
Graph Propagation
     │
     ▼
New Contextual State
```

This provides the project's current interpretation of Attention:

> **Attention is state-dependent graph propagation.**

---

## State Machine Perspective

The Transformer can also be viewed as a state transition system.

Let the current token state be

$$
S_t:D\times N\rightarrow\mathbb{R}.
$$

Attention derives a graph from this state:

$$
A(S_t)
=
\operatorname{softmax}
\left(
\frac{
(W_QS_t)^T(W_KS_t)
}{
\sqrt D
}
\right).
$$

The value transformation gives:

$$
M_t=W_VS_t.
$$

Information is then propagated through the dynamically constructed graph:

$$
Y_t=M_tA(S_t).
$$

With residual connections, the process can be expressed conceptually as:

$$
S_{t+1}
=
S_t+\Delta(S_t).
$$

This creates a direct bridge between:

* state machines,
* matrix operations,
* graph propagation,
* and Transformer blocks.

---

## Why Pure Python?

The early stages intentionally avoid PyTorch and other deep-learning frameworks.

The purpose is not performance.

The purpose is visibility.

A framework can express:

```python
attention(...)
```

with only a few lines of code.

This project instead asks:

* What is a vector?
* What is a matrix?
* Which index is being summed?
* Why does matrix multiplication have this shape?
* Why does \(Q^TK\) produce an \(N\times N\) matrix?
* Why is softmax applied across a particular dimension?
* What does \(W_V\) actually change?
* Where does the graph appear?
* How does the state change?

Only after these structures are understood does a higher-level framework become useful.

PyTorch and automatic differentiation can be introduced later when the project reaches training.

---

## Current Progress

### Completed

* [x] Character sequence representation
* [x] Byte Pair Encoding (BPE)
* [x] Symbol sequence generation
* [x] Co-occurrence graph
* [x] Sparse co-occurrence matrix
* [x] Vector representation
* [x] Matrix representation
* [x] Matrix-vector multiplication
* [x] Matrix-matrix multiplication
* [x] Matrix transpose
* [x] Co-occurrence embedding
* [ ] SVD / power iteration foundations
* [x] Token state representation
* [x] Scaled dot-product Attention
* [x] Softmax
* [x] Attention graph propagation

### Current Milestone

> **Attention — Completed**

The Attention implementation has been verified with deterministic tests and is currently the main completed Transformer component.

### Next

* [ ] Residual connections
* [ ] Layer Normalization
* [ ] Feed-Forward Network
* [ ] Transformer Block
* [ ] Multi-Head Attention
* [ ] Positional representation
* [ ] Training / automatic differentiation
* [ ] Language modeling
* [ ] Tiny GPT-style model

---

## Project Structure

```text
math-transformer/
├── README.md
├── README.zh-CN.md
├── docs/
│   ├── tokenizer/
│   │   └── 1.tokenizer.md
│   ├── embedding/
│   │   └── 2.embedding.md
│   └── transformer/
│       └── 3.attention.md
│
├── src/
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── bpe.py
│   │   └── ngram.py
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── co_occurrence_embedding.py
│   │   ├── power_iteration.py
│   │   ├── sparse_matrix.py
│   │   ├── svd.py
│   │   └── vector.py
│   │
│   └── transformer/
│   │   ├── transformer.py
│
└── tests/
    └── ...
```

The structure follows the conceptual progression of the project rather than the architecture of an existing deep-learning framework.

---

## Learning Philosophy

The project follows a simple principle:

> **Understand the mathematical structure before hiding it behind an abstraction.**

Rather than starting from an existing Transformer implementation and working backward, the project builds upward:

```text
Real Analysis
      ↓
Functions
      ↓
Finite-dimensional Functions
      ↓
Vectors
      ↓
Matrices
      ↓
Graphs
      ↓
Embeddings
      ↓
Dynamic Graphs
      ↓
Attention
      ↓
State Transitions
      ↓
Transformer
```

The goal is not to minimize the amount of code.

The goal is to minimize the amount of **unexplained structure**.

---

## Relationship Between the Stages

The project can be summarized through two different graph perspectives.

### Static representation

```text
Corpus
  │
  ▼
Symbols
  │
  ▼
Co-occurrence
  │
  ▼
Static Graph
  │
  ▼
Embedding
  │
  ▼
Token Vectors
```

The relationships are extracted from the corpus and remain fixed after construction.

### Dynamic representation

```text
Token States
     │
     ▼
   Q / K
     │
     ▼
Dynamic Graph
     │
     ▼
 Attention Weights
     │
     ▼
 V Transformation
     │
     ▼
Graph Propagation
     │
     ▼
Contextual States
```

The relationships themselves depend on the current state.

This transition from a static graph to a state-dependent graph is one of the central ideas of the project.

---

## Corpus

The current experiments use **Journey to the West (西游记)** as the primary corpus.

The corpus is used as a concrete environment for exploring:

* tokenization,
* symbol relationships,
* embedding,
* contextual representation,
* and eventually language modeling.

The corpus itself is not the primary subject of the project.

It is the data through which the mathematical structures become observable.

---

## Design Principles

### 1. Mathematics before abstraction

Every important operation should first be understandable as a mathematical operation.

### 2. Explicit dimensions

Shapes are treated as part of the mathematical meaning of an operation.

For example:

$$
X:D\times N
$$

is not merely a storage detail.

It says that every sequence position has a \(D\)-dimensional state.

### 3. Index-oriented reasoning

Rather than relying only on geometric intuition, operations are understood by fixing output indices and identifying the index being summed or propagated.

### 4. State over static data

A Transformer is not treated as a collection of matrix operations.

It is viewed as a mechanism that repeatedly transforms a state.

### 5. Graph interpretation

Attention is understood as dynamically constructing and propagating over a graph.

### 6. Frameworks later

High-level frameworks are useful for optimization, automatic differentiation, and large-scale computation.

They are intentionally introduced after the mathematical structure becomes clear.

---

## Status

This project is under active development.

The implementation is intentionally incremental: each stage should be mathematically understood, implemented, and tested before moving to the next abstraction layer.

Current position:

$$
\boxed{
\text{Attention}
\;\checkmark
}
$$

Next major milestone:

$$
\boxed{
\text{Transformer Block}
}
$$

---

## License

This project is primarily a personal learning and experimentation project.
