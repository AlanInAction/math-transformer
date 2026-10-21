import torch
import torch.nn as nn
import logging

from src.transformer.model import ForwardOnlyTransformer
from src.transformer.transformer import Tokenizer


logger = logging.getLogger(__name__)


def create_training_sample(token_ids, seq_len):
    tokens = torch.tensor(token_ids, dtype=torch.long)

    input_ids = tokens[:seq_len]
    targets = tokens[1:seq_len + 1]

    return input_ids, targets

def get_batch(token_ids, batch_size, seq_len, device="cpu"):
    max_start = len(token_ids) - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))

    x = torch.stack([
        torch.tensor(token_ids[s:s + seq_len])
        for s in starts.tolist()
    ])

    y = torch.stack([
        torch.tensor(token_ids[s + 1:s + seq_len + 1])
        for s in starts.tolist()
    ])

    return x.to(device), y.to(device)

if __name__=="__main__":

    logging.basicConfig(
        level=logging.DEBUG 
    )
    path = "/Users/alan/dev/refactor/math-lab/corpus/西游记.txt"
    n_limit = 50000
    ite_limit = 1000
    window_size = 4 # window contains symbol_i
    bidirection = True

    bpe_tokenizer = Tokenizer(
        n_limit=50000,
        ite_limit=1000,
        window_size=4,
        bidirection=True,
    )

    bpe_tokenizer.tokenize(path)
    token_id_map, sparse_co_occurrence_map, dimension = (
        bpe_tokenizer.get_embedding()
    )

    symbol_sequence = bpe_tokenizer.symbol_sequence

    token_ids = [
        token_id_map[symbol]
        for symbol in symbol_sequence
    ]

    # assert len(token_ids) > 1
    # assert all(0 <= token_id < dimension for token_id in token_ids)

    logger.info("vocab_size: %d", dimension)
    logger.info("sequence_length: %d", len(token_ids))


    # model = ForwardOnlyTransformer(
    #     vocab_size=dimension,
    #     d_model=64,
    #     d_ff=128,
    #     max_seq_len=128,
    #     n_layers=2,
    # )

    # seq_len = 64
    # input_ids, targets = create_training_sample(token_ids, seq_len)

    # # 添加 Batch 维度
    # input_ids = input_ids.unsqueeze(0)
    # targets = targets.unsqueeze(0)

    # logits = model(input_ids)

    # assert logits.shape == (1, seq_len, dimension)

    vocab_size = dimension
    seq_len = 64

    model = ForwardOnlyTransformer(
        vocab_size=vocab_size,
        d_model=64,
        d_ff=128,
        max_seq_len=128,
        n_layers=2,
    )

    # 序列长度 [0,65)
    tokens = torch.tensor(token_ids[:seq_len + 1], dtype=torch.long)

    input_ids = tokens[:-1].unsqueeze(0)
    targets = tokens[1:].unsqueeze(0)

    logits = model(input_ids)

    assert logits.shape == (1, seq_len, vocab_size)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

    print("input_ids shape:", input_ids.shape)
    print("targets shape:", targets.shape)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 单次训练
    with torch.no_grad():
        logits_after = model(input_ids)
        loss_after = loss_fn(
            logits_after.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

    print("loss before:", loss.item())
    print("loss after:", loss_after.item())


    num_steps = 31
    batch_size = 10
    for step in range(num_steps):
        input_ids, targets = get_batch(
            token_ids, batch_size, seq_len
        )

        logits = model(input_ids)
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(step, loss.item())

    from src.transformer.generate import generate

    # 选择一段已有的 Token 序列作为提示
    prompt_symbols = ["孙悟空", "在", "花果山"]

    prompt_ids = [
        token_id_map[symbol]
        for symbol in prompt_symbols
        if symbol in token_id_map
    ]

    assert prompt_ids, "提示词没有匹配到词表中的 Token"

    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_ids = generate(
        model,
        input_ids,
        max_new_tokens=20,
    )

    reverse_id_token_map = dict()
    for k,v in token_id_map.items():
        reverse_id_token_map[v]=k
    reverse_id_token_map = reverse_id_token_map
    # 将生成的 Token ID 还原为 BPE 符号
    generated_symbols = [
        reverse_id_token_map[token_id]
        for token_id in generated_ids[0].tolist()
    ]

    print("Generated symbols:", generated_symbols)
    print("Generated text:", "".join(generated_symbols))


    next_token_logits = logits[:, -1, :]
    top_values, top_indices = torch.topk(
        next_token_logits, k=5, dim=-1
    )

    top_probs = torch.softmax(top_values, dim=-1)

    for token_id, prob in zip(
        top_indices[0].tolist(),
        top_probs[0].tolist(),
    ):
        print(reverse_id_token_map[token_id], prob)

    probs = torch.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=5, dim=-1)