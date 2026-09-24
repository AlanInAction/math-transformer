# math-transformer

从**实分析、线性代数、向量、矩阵与图传播**的角度探索 Transformer，并使用纯 Python 从零实现。

[English README](README.md)

---

## 项目简介

`math-transformer` 是一个以数学理解为核心、从第一性原理逐步构建 Transformer 的学习型项目。

项目的目标并不是简单地复现一个 Transformer，而是沿着一条连续的数学路径，理解一个 Transformer 是如何逐步从语料、符号、向量和矩阵中构建出来的：

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

项目早期阶段刻意使用**纯 Python**实现，而不是直接依赖 PyTorch 等深度学习框架。

这样做的目的不是追求性能，而是让底层的数学运算保持可见，使每一个实现步骤都能够与对应的数学结构建立联系。

---

## 核心思想

整个项目建立在三个相互联系的视角之上：

```text
                 ┌──────────────┐
                 │   State      │
                 │    Machine   │
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

这里并不把它们看作三个独立的知识点。

Transformer 可以被理解为一个**状态转移系统**：

* 状态由向量、矩阵表示；
* Attention 根据当前状态动态构造图；
* 信息沿着这个动态图传播；
* 最终得到新的上下文状态。

因此，项目中的一个核心抽象是：

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

## 数学视角

项目中的一个重要出发点，是把向量理解为一个有限维函数。

对于索引集 \(I\)：

$$
V:I\rightarrow\mathbb{R}
$$

向量本质上是一个以索引为自变量、以实数为取值的函数。

例如：

$$
I=\{0,1,2\}
$$

则：

$$
V:I\rightarrow\mathbb{R}
$$

可以表示为：

$$
V=(v_0,v_1,v_2).
$$

在这个视角下，矩阵自然可以推广为两个索引集上的函数：

$$
M:I\times J\rightarrow\mathbb{R}.
$$

于是，向量、矩阵以及更高维的 Tensor，都可以统一理解为定义在不同乘积索引集上的函数。

### 矩阵乘法

对于：

$$
A:I\times J\rightarrow\mathbb{R}
$$

以及：

$$
B:J\times K\rightarrow\mathbb{R},
$$

矩阵乘法得到：

$$
AB:I\times K\rightarrow\mathbb{R},
$$

其中：

$$
(AB)(i,k)
=
\sum_{j\in J}A(i,j)B(j,k).
$$

也就是说：

> 固定输出索引，对共享索引进行求和。

这种理解方式在理解 Attention 时尤其重要。

---

## 从静态图到动态图

Tokenizer 和 Embedding 阶段首先从语料中建立符号之间的关系。

共现关系可以表示成一个带权图：

$$
G_{\text{static}}=\langle V,E\rangle
$$

其中：

* \(V\) 表示 Token / Symbol；
* \(E\) 表示共现关系；
* 边的权重表示共现关系的强弱。

Embedding 再将这些符号之间的关系映射到向量空间。

Transformer 则引入了另一种图。

给定当前的 Token State：

$$
X:D\times N\rightarrow\mathbb{R},
$$

其中：

* \(D\) 是 Embedding / State 的维度；
* \(N\) 是 Sequence 的索引集；

Attention 会根据当前状态构造一个动态图。

因此：

$$
A=A(X).
$$

这里的图不再是固定的。

**图本身会随着当前状态发生变化。**

---

## Attention

当前 Attention 实现遵循 Scaled Dot-Product Attention 的数学结构。

给定：

$$
X\in\mathbb{R}^{D\times N},
$$

首先通过三个独立的参数变换得到：

$$
Q=W_QX,
$$

$$
K=W_KX,
$$

$$
V=W_VX.
$$

然后计算不同 Sequence Position 之间的关系：

$$
S=Q^TK.
$$

因为：

$$
Q^T\in\mathbb{R}^{N\times D},
\qquad
K\in\mathbb{R}^{D\times N},
$$

所以：

$$
S\in\mathbb{R}^{N\times N}.
$$

这个 \(N\times N\) 的矩阵可以理解为一个作用在 Sequence 上的**动态图**。

经过缩放：

$$
\hat S=\frac{Q^TK}{\sqrt D},
$$

再通过 Softmax 得到归一化的权重：

$$
A=\mathrm{softmax}(\hat S).
$$

最后使用这些权重传播经过 \(W_V\) 变换后的 Token State：

$$
Y=VA.
$$

因此整个过程可以表示为：

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
\mathrm{softmax}
\rightarrow
VA
}
$$

从图的角度来看：

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

因此，当前项目对 Attention 的核心理解可以概括为：

> **Attention 是一种依赖当前状态的图传播机制。**

---

## State Machine 视角

Transformer 也可以被理解为一个状态转移系统。

设当前 Token State 为：

$$
S_t:D\times N\rightarrow\mathbb{R}.
$$

Attention 根据当前状态构造动态图：

$$
A(S_t)
=
\mathrm{softmax}
\left(
\frac{
(W_QS_t)^T(W_KS_t)
}{
\sqrt D
}
\right).
$$

Value Transformation：

$$
M_t=W_VS_t.
$$

随后信息沿着动态图进行传播：

$$
Y_t=M_tA(S_t).
$$

加入 Residual Connection 后，可以进一步抽象为：

$$
S_{t+1}
=
S_t+\Delta(S_t).
$$

于是，Attention、State Machine、Matrix Operation 和 Graph Propagation 之间建立起了直接联系。

---

## 为什么使用纯 Python？

项目早期阶段刻意不使用 PyTorch 或其他深度学习框架。

目的不是性能。

而是**可见性**。

一个框架可以用非常少的代码表达：

```python
attention(...)
```

但项目真正想理解的是：

* Vector 到底是什么？
* Matrix 到底是什么？
* Matrix Multiplication 究竟在哪个 Index 上求和？
* 为什么矩阵乘法具有这样的 Shape？
* 为什么 \(Q^TK\) 得到的是 \(N\times N\)？
* Softmax 为什么沿着特定维度计算？
* \(W_V\) 到底改变了什么？
* Graph 在哪里出现？
* State 又是如何发生变化的？

因此，项目首先尝试把这些结构直接实现出来。

只有当数学结构已经清晰之后，再引入 PyTorch、Automatic Differentiation 等高级抽象。

---

## 当前进度

### 已完成

* [x] Character Sequence
* [x] Byte Pair Encoding (BPE)
* [x] Symbol Sequence
* [x] Co-occurrence Graph
* [x] Sparse Co-occurrence Matrix
* [x] Vector Representation
* [x] Matrix Representation
* [x] Matrix-Vector Multiplication
* [x] Matrix-Matrix Multiplication
* [x] Matrix Transpose
* [x] Co-occurrence Embedding
* [x] SVD / Power Iteration 基础
* [x] Token State Representation
* [x] Scaled Dot-Product Attention
* [x] Softmax
* [x] Attention Graph Propagation

### 当前里程碑

> **Attention — Completed**

Attention 已经完成纯 Python 实现，并通过确定性测试验证。

这是目前 Transformer 部分的第一个完整数学与工程里程碑。

### 下一阶段

* [ ] Residual Connection
* [ ] Layer Normalization
* [ ] Feed-Forward Network
* [ ] Transformer Block
* [ ] Multi-Head Attention
* [ ] Positional Representation
* [ ] Training / Automatic Differentiation
* [ ] Language Modeling
* [ ] Tiny GPT-style Model

---

## 项目结构

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
│       └── ...
│
└── tests/
    └── ...
```

项目结构本身也遵循从数学结构逐渐走向 Transformer 的路线，而不是直接复制某个深度学习框架的目录结构。

---

## 学习理念

项目遵循一个简单的原则：

> **在使用抽象隐藏细节之前，先理解抽象背后的数学结构。**

与其从一个现成的 Transformer 实现开始，再反过来理解它，不如从基础结构逐步向上构建：

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

这里并不追求尽可能少的代码。

真正希望减少的是：

> **没有被解释清楚的结构。**

---

## 各阶段之间的关系

项目目前可以通过两种不同的图视角进行概括。

### 静态表示

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

这些关系从语料中提取出来，在图构建完成之后保持相对固定。

### 动态表示

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

这里的关系本身依赖于当前的状态。

因此，项目的一条核心演化路线是：

$$
\boxed{
\text{Static Graph}
\rightarrow
\text{Dynamic Graph}
}
$$

也可以理解为：

$$
\boxed{
\text{Fixed Relationship}
\rightarrow
\text{State-dependent Relationship}
}
$$

---

## 语料

当前实验主要使用**《西游记》**作为语料。

语料被用于观察和实验：

* Tokenization
* Symbol Relationship
* Embedding
* Contextual Representation
* Language Modeling

语料本身并不是项目研究的重点。

它更像是一个具体的实验环境，让这些数学结构能够从抽象概念变成可以观察、实现和验证的对象。

---

## 设计原则

### 1. 数学优先于抽象

重要操作首先应该能够被理解为数学运算。

### 2. 显式理解维度

Shape 不只是数据存储的细节，而是数学对象的一部分。

例如：

$$
X:D\times N
$$

它表达的是：

> 每一个 Sequence Position 都具有一个 \(D\) 维状态。

### 3. Index-oriented Reasoning

不只依赖几何直觉，而是通过固定输出 Index，寻找被求和、变换或传播的 Index 来理解运算。

### 4. State 优先于静态数据

Transformer 不被理解为一组孤立的矩阵运算。

它被理解为一个不断更新 State 的过程。

### 5. Graph Interpretation

Attention 被理解为动态构造 Graph，并沿 Graph 进行信息传播。

### 6. Framework Later

高级框架最终会用于：

* Automatic Differentiation
* Training
* Large-scale Computation
* Performance Optimization

但它们被有意放在数学结构理解之后。

---

## 当前状态

项目仍在持续开发。

整个实现过程遵循一个渐进原则：

> **每一个阶段先理解数学结构，再实现，再测试，然后进入下一层抽象。**

当前所在位置：

$$
\boxed{
\text{Attention}
\;\checkmark
}
$$

下一阶段：

$$
\boxed{
\text{Transformer Block}
}
$$

---

## License

This project is primarily a personal learning and experimentation project.
