import torch

from src.transformer.model import (
    CausalSelfAttention,
    FeedForward,
    TransformerBlock,
    ForwardOnlyTransformer,
)


def test_attention_shape():
    B, T, D = 2, 8, 32

    model = CausalSelfAttention(D)
    x = torch.randn(B, T, D)

    output = model(x)

    assert output.shape == (B, T, D)


def test_feedforward_shape():
    B, T, D = 2, 8, 32

    model = FeedForward(D, d_ff=64)
    x = torch.randn(B, T, D)

    output = model(x)

    assert output.shape == (B, T, D)


def test_transformer_block_shape():
    B, T, D = 2, 8, 32

    model = TransformerBlock(D, d_ff=64)
    x = torch.randn(B, T, D)

    output = model(x)

    assert output.shape == (B, T, D)


def test_model_logits_shape():
    B, T = 2, 8
    vocab_size = 100

    model = ForwardOnlyTransformer(
        vocab_size=vocab_size,
        d_model=32,
        d_ff=64,
        max_seq_len=16,
        n_layers=2,
    )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(B, T),
    )

    logits = model(input_ids)

    assert logits.shape == (B, T, vocab_size)


def test_attention_is_causal():
    B, T, D = 1, 5, 8
    model = CausalSelfAttention(D)
    model.eval()

    x1 = torch.randn(B, T, D)
    x2 = x1.clone()

    # 1: 修改 x2 的未来位置，例如位置 3、4
    x2[:, 3:, :] = torch.randn_like(x2[:, 3:, :])

    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)

    # 2: 比较未来位置之前的输出 shape(-1)>3
    assert torch.allclose(y1[:, :3, :], y2[:, :3, :],rtol=1e-5)
   
def test_transformer_is_causal():
    B, T = 1, 5
    vocab_size = 32

    model = ForwardOnlyTransformer(
        vocab_size=vocab_size,
        d_model=16,
        d_ff=32,
        max_seq_len=8,
        n_layers=2,
    )
    model.eval()

    x1 = torch.randint(0, vocab_size, (B, T))
    x2 = x1.clone()

    # 修改未来位置 3、4 的 token
    x2[:, 3:] = torch.randint(
        0, vocab_size, x2[:, 3:].shape
    )

    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)

    # 过去位置的所有词表 logits 都应该保持不变
    assert torch.allclose(
        y1[:, :3, :],
        y2[:, :3, :],
        atol=1e-6,
        rtol=1e-5,
    )


if __name__ == "__main__":
    test_attention_shape()
    test_feedforward_shape()
    test_transformer_block_shape()
    test_attention_is_causal()
    test_model_logits_shape()
    test_transformer_is_causal()
    
    print("All forward-only tests passed.")