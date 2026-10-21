
import torch


@torch.no_grad()
def generate(model, input_ids, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):
        # 保留模型上下文窗口内最近的 Token
        input_context = input_ids[:, -model.max_seq_len:]

        # 前向计算，取最后一个位置的预测分数
        logits = model(input_context)
        next_token_logits = logits[:, -1, :]

        # 当前先采用贪心解码：选择分数最高的 Token
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # 将预测的 Token 追加到已有序列
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    return input_ids
