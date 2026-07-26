import torch
from einops import rearrange
from jaxtyping import Int

from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import TransformerLM


def decode(
    lm: TransformerLM,
    x: Int[torch.Tensor, "... seq_len"],
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> Int[torch.Tensor, "... out_len"]:
    finished = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)

    for _ in range(max_tokens):
        model_input = x[..., -lm.context_length :]
        y = lm(model_input)
        probabilities = softmax(y[..., -1, :] / temperature, dim=-1)

        if top_p is not None:
            probabilities, token_ids = torch.sort(probabilities, dim=-1, descending=True)
            cumulative_probabilities = torch.cumsum(probabilities, dim=-1)
            probabilities = probabilities.masked_fill(cumulative_probabilities - probabilities >= top_p, 0)
            probabilities /= probabilities.sum(dim=-1, keepdim=True)
            sampled_rank = torch.multinomial(probabilities, num_samples=1)
            y_next = torch.gather(token_ids, dim=-1, index=sampled_rank)
        else:
            y_next = torch.multinomial(probabilities, num_samples=1)

        if eos_token_id is not None:
            y_next = y_next.masked_fill(finished[..., None], eos_token_id)

        x = torch.cat([x, y_next], dim=-1)

        if eos_token_id is not None:
            finished |= rearrange(y_next, "... 1 -> ...") == eos_token_id
            if finished.all():
                break

    return x


if __name__ == "__main__":
    from pathlib import Path

    from cs336_basics.checkpointing import load_config
    from cs336_basics.tokenizer import Tokenizer

    checkpoint = sorted(Path("data/checkpoints").glob("*/*.pt"))[-5]
    config = load_config(checkpoint)
    model = TransformerLM(**config)
    state = torch.load(checkpoint, map_location=model.device)
    model.load_state_dict(state["model"])
    model.eval()

    tokenizer = Tokenizer.from_files(
        "data/TinyStoriesV2-GPT4-train.txt.bpe.vocab.json",
        "data/TinyStoriesV2-GPT4-train.txt.bpe.merges.json",
        special_tokens=["<|endoftext|>"],
    )
    x = torch.tensor([tokenizer.encode("Once upon a time")], dtype=torch.long, device=model.device)
    eos_token_id = tokenizer.re_vocab[b"<|endoftext|>"]
    out = decode(model, x, max_tokens=256, temperature=0.8, eos_token_id=eos_token_id)
    print(tokenizer.decode(out[0].tolist()))
