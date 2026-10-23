import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """单头因果自注意力。"""

    def __init__(self, d_model: int):
        super().__init__()
        # 加权重的内积,nn.Linear将权重存储为doutxdin
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # 通过nxn的权重矩阵换出dxn
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            Batch
            Token Length
            Dimension
        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape

        # 1: 计算 Q、K、V
        # dxd btd
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2: 计算 Attention scores
        # 目标形状: (B, T, T) BTD BDT
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(D)

        # 3: 构造 causal mask
        # mask 中 True 表示不允许访问的位置
        # 目标形状: (T, T)  triu triangular
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        # 4: 将不允许访问的位置设为负无穷
        scores = scores.masked_fill(mask, float("-inf"))

        # 5: Softmax 得到 Attention weights
        weights = torch.softmax(scores, dim=-1)

        # 6: 使用 weights 聚合 V
        # 目标形状: (B, T, D) BTT BTD
        output = weights @ V

        # 7: 输出投影 BTD DD BTD
        output = self.w_o(output)  # output.transpose(-2,-1)

        return output


class FeedForward(nn.Module):
    """逐 Token 的前馈网络。
    整体是带非线性单元的MLP Multiple Layer Perceptron
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8: 前向计算
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个简化的 Transformer Decoder Block。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.attention = CausalSelfAttention(d_model)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 9: Attention、Residual、LayerNorm
        attention_delta = self.attention(x)
        update_val_1 = x + attention_delta
        norm_val_1 = self.norm1(update_val_1)

        # 10: FFN、Residual、LayerNorm
        ffn_delta = self.ffn(norm_val_1)
        update_val_2 = norm_val_1 + ffn_delta
        norm_val_2 = self.norm2(update_val_2)

        return norm_val_2


class ForwardOnlyTransformer(nn.Module):
    """用于验证前向计算的简化 GPT Decoder。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        d_ff: int = 128,
        max_seq_len: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_ff) for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T)

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # 11: 检查序列长度不超过 max_seq_len    input_ids.shape()[-1]
        if T > self.max_seq_len:
            raise ValueError("can't perform forward,the input shape beyond max_seq_len!")

        # 12: 构造位置索引 (T,)
        positions_shape = torch.arange(T, device=input_ids.device)
        # position输入为T,输出为T,D
        position_embedding_val = self.position_embedding(positions_shape)

        # 13: Token Embedding + Position Embedding
        # token embedding 输入B,T，输出B,T,D
        token_embedding_val = self.token_embedding(input_ids)
        x = position_embedding_val + token_embedding_val

        # TODO 14: 依次通过 Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 15: 最终归一化
        norm_val = self.final_norm(x)

        # 16: 映射到词表维度
        # 输入 B,T,D 输出B,T,V lm_head目前是(out,in) (V,D)，实际矩阵乘法内部实现是：转置后是D,V
        logits = self.lm_head(norm_val)

        return logits
