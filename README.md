# TinyGPT Learning Project Roadmap

## 1. 项目目标

通过从零实现一个最小化 GPT 模型，理解现代大语言模型的核心机制：

* 自然语言如何转换为向量空间
* Embedding 如何学习语义表示
* Attention 如何动态建模上下文关系
* Transformer 如何通过函数复合形成复杂表示
* 大模型如何通过梯度下降学习语言分布

最终目标：

> 从零实现一个能够完成 Next Token Prediction 的 TinyGPT，并理解其内部数学结构。

---

# 2. 总体学习路线

```
形式语言
    |
    v
Tokenizer
    |
    v
Embedding
    |
    v
Word2Vec
    |
    v
Attention
    |
    v
Transformer Block
    |
    v
Decoder-only GPT
    |
    v
TinyGPT Training
```

对应数学路线：

```
离散集合
    |
    v
映射
    |
    v
向量空间
    |
    v
线性变换
    |
    v
概率分布
    |
    v
优化
    |
    v
函数逼近
```

---

# 3. Milestone Roadmap

| Milestone | 目标                | 核心概念    | 数学对应                  | 实现内容                         | 验收标准                  |
| --------- | ----------------- | ------- | --------------------- | ---------------------------- | --------------------- |
| M0        | 项目初始化             | AI实验环境  | 函数实验空间                | Python/Numpy/PyTorch环境、项目结构  | 可以运行实验代码              |
| M1        | Tokenizer         | 离散语言表示  | 集合、映射                 | 文本与token id转换                | 完成tokenize/detokenize |
| M2        | Embedding         | 符号到向量   | 映射 (X\rightarrow R^n) | 手写Embedding Layer            | token得到向量             |
| M3        | Word2Vec          | 学习语义空间  | 概率模型、梯度下降             | Skip-Gram训练                  | 相似语义距离接近              |
| M4        | Attention基础       | 理解Q/K/V | 线性变换、内积空间             | NumPy实现Single Head Attention | 理解attention计算         |
| M5        | Attention训练       | 学习关系    | 参数优化                  | 训练Q/K/V矩阵                    | Attention产生语义关系       |
| M6        | Position Encoding | 引入顺序信息  | Fourier Basis         | 实现sin/cos位置编码                | 模型感知位置                |
| M7        | Transformer Block | 函数复合    | 高维函数组合                | Attention + FFN + LayerNorm  | 单层Transformer运行       |
| M8        | GPT Decoder       | 自回归生成   | 条件概率                  | Causal Mask + LM Head        | 完成next token预测        |
| M9        | Training Pipeline | 完整训练闭环  | 优化理论                  | Dataset/Loss/Optimizer       | Loss下降                |
| M10       | TinyGPT训练         | 产生语言能力  | 函数逼近                  | 小语料训练                        | 可以生成简单文本              |
| M11       | 可解释分析             | 理解模型内部  | 几何空间                  | Embedding/Attention可视化       | 观察语义结构                |
| M12       | PyTorch重构         | 工业实现    | 自动微分                  | nn.Module版本                  | 接近真实LLM结构             |

---

# 4. 核心模块设计

## 4.1 Tokenizer

目标：

将自然语言映射为离散符号。

输入：

```
我喜欢苹果
```

输出：

```
[12,53,89]
```

核心：

建立：

```
token <-> id
```

对应：

形式语言中的：

[
\Sigma
]

---

# 4.2 Embedding

目标：

将离散空间转换为连续向量空间。

数学：

[
f:
Vocabulary
\rightarrow
R^d
]

实现：

Embedding矩阵：

[
E\in R^{V\times d}
]

查询：

[
x_i=E[i]
]

理解：

Embedding本质：

> 一个可学习的高维坐标系统。

---

# 4.3 Word2Vec

目标：

让模型学习词语之间的语义关系。

训练方式：

输入：

```
king
```

预测：

```
queen
man
woman
```

优化：

[
P(context|word)
]

最终：

语义关系转化为向量空间关系。

例如：

[
king-man+woman\approx queen
]

---

# 4.4 Attention

核心公式：

[
Attention(Q,K,V)
================

softmax(
\frac{QK^T}{\sqrt d}
)V
]

其中：

输入：

[
X
]

生成：

[
Q=XW_Q
]

[
K=XW_K
]

[
V=XW_V
]

理解：

* Q：当前需要寻找的信息
* K：当前token提供的索引特征
* V：实际语义内容

Attention本质：

> 根据输入动态生成一个关系矩阵，并重新组合信息。

---

# 4.5 Position Encoding

Transformer没有递归结构，因此需要额外加入位置信息。

经典形式：

[
PE(pos,2i)
==========

sin(pos/10000^{2i/d})
]

[
PE(pos,2i+1)
============

cos(pos/10000^{2i/d})
]

对应：

Fourier basis思想。

---

# 4.6 Transformer Block

结构：

```
Input

 |
Embedding

 |
Self Attention

 |
Feed Forward

 |
Residual

 |
LayerNorm

 |
Output
```

数学：

[
F(x)=f_n\circ f_{n-1}...\circ f_1(x)
]

本质：

高维函数复合。

---

# 4.7 GPT Decoder

目标：

预测：

[
P(x_t|x_1,x_2,...,x_{t-1})
]

增加：

Causal Mask

保证：

当前token不能看到未来信息。

---

# 5. 项目目录建议

```
tinygpt/

├── README.md
│
├── docs/
│   ├── embedding.md
│   ├── attention.md
│   ├── transformer.md
│   └── math_notes.md
│
├── data/
│
├── tokenizer/
│   └── tokenizer.py
│
├── model/
│   ├── embedding.py
│   ├── attention.py
│   ├── transformer.py
│   └── loss.py
│
├── optimizer/
│
├── train/
│   ├── dataset.py
│   └── trainer.py
│
├── inference/
│
└── experiments/
    ├── embedding_visualization.ipynb
    └── attention_visualization.ipynb
```

---

# 6. 推荐实验顺序

## Experiment 1

### Embedding空间实验

观察：

```
cat
dog
lion

apple
banana
orange
```

训练后是否自动聚集。

---

## Experiment 2

### Attention Matrix可视化

观察：

```
        I love apple

I       0.8 0.1 0.1

love    0.2 0.7 0.1

apple   0.1 0.2 0.7
```

理解：

模型如何选择上下文。

---

## Experiment 3

### TinyGPT生成实验

观察：

随机初始化：

```
完全随机输出
```

训练后：

```
出现语法结构
```

理解：

梯度下降如何让：

[
F_\theta
]

逼近：

[
P(language)
]

---

# 7. 数学与AI对应关系

| 数学概念 | TinyGPT对应         |
| ---- | ----------------- |
| 集合   | Vocabulary        |
| 离散空间 | Token             |
| 映射   | Embedding         |
| 向量空间 | Feature Space     |
| 线性变换 | Q/K/V Projection  |
| 矩阵乘法 | Attention         |
| 概率分布 | Softmax           |
| 交叉熵  | Loss              |
| 导数   | Gradient          |
| 优化   | Parameter Update  |
| 函数复合 | Transformer Block |
| 函数逼近 | Language Model    |

---

# 8. 后续扩展方向

完成 TinyGPT 后：

进入：

## TinyCodingAgent

结构：

```
TinyGPT

+

Tool Calling

+

State Machine

+

Memory

+

Feedback Loop
```

对应：

```
LLM
 |
Agent Controller
 |
Tools
 |
Environment
 |
Observation
```

最终形成：

AI模型 → Agent系统 → 软件工程自动化

---

# 9. 项目原则

1. 优先理解数学机制，而不是调用框架。
2. 每个模块都有最小可运行实现。
3. 每个公式都对应代码。
4. 每个模型能力都通过实验验证。
5. 保留实验记录和数学推导。

最终目标：

> 构建一个从数学理论、模型原理到工程应用完整贯通的个人AI工程体系。

```
```

这个文档可以直接作为仓库第一版设计文档。后续每完成一个 milestone，可以继续在对应目录增加实验记录和数学推导。
