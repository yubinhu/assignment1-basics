"""Train a Transformer LM. Example:

    uv run python -m cs336_basics.training_together \
        --train data/tinystories_train_ids.npy --valid data/tinystories_valid_ids.npy \
        --vocab-size 10000 --max-iters 5000

Checkpoints land in data/checkpoints/<train stem>-<timestamp>/<iteration>.pt. To pick a run back
up, point --resume-from at that directory (latest iteration wins) or at a specific .pt file; the
model config travels with the checkpoint, so no model flags need repeating.

All hyperparameter defaults live on `train_llm`, so it is equally usable as a library call.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, load_config, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import load_data
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import cosine_learning_rate_schedule
from cs336_basics.transformer_lm import TransformerLM

CHECKPOINT_DIR = Path("data/checkpoints")

# The kwargs TransformerLM is built from; these travel with the checkpoint so a resumed
# run rebuilds the same architecture without the caller re-passing every flag.
MODEL_KEYS = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")


def resolve_checkpoint(resume_from: Path) -> Path:
    """Accept either a checkpoint file or a run directory, in which case take the latest iteration.

    Filenames are zero-padded, so lexical order matches numeric order.
    """
    if not resume_from.is_dir():
        return resume_from
    checkpoints = sorted(resume_from.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no .pt checkpoints in {resume_from}")
    return checkpoints[-1]


def open_dataset(path: Path, vocab_size: int) -> np.ndarray:
    """Memory-map the token IDs so only the sampled windows are ever read off disk."""
    data = np.load(path, mmap_mode="r")
    assert data.ndim == 1, f"{path} should be a flat array of token IDs, got shape {data.shape}"
    assert data[:1000].max() < vocab_size, f"{path} has token IDs beyond vocab_size={vocab_size}"
    return data


def compute_loss(model: TransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """cross_entropy expects 2D logits and 1D targets, so fold batch and position together."""
    logits = model(x)
    return cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = [
        compute_loss(model, *load_data(data, batch_size, context_length, device)).item()
        for _ in range(eval_iters)
    ]
    model.train()
    return sum(losses) / len(losses)


def train_llm(
    train: Path,
    valid: Path,
    *,
    # model
    vocab_size: int = 10_000,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int | None = None,  # PositionwiseFeedForward's default, round(8/3 * d_model / 64) * 64
    rope_theta: float = 10_000.0,  # NOT a redundant default: None disables RoPE entirely
    # optimization
    batch_size: int = 32,
    max_iters: int = 5_000,
    lr_max: float = 1e-3,
    lr_min: float = 1e-4,
    warmup_iters: int = 200,
    cosine_cycle_iters: int | None = None,  # defaults to max_iters
    weight_decay: float = 0.01,
    betas: tuple[float, float] | None = None,  # AdamW's default
    eps: float | None = None,  # AdamW's default
    max_grad_norm: float = 1.0,
    # logging, checkpointing, hardware
    log_every: int = 100,
    eval_every: int = 500,
    eval_iters: int = 20,
    checkpoint_every: int = 1_000,
    checkpoint_dir: Path | None = None,  # defaults to data/checkpoints/<train stem>-<timestamp>/
    resume_from: Path | None = None,  # a checkpoint file, or a run directory to take the latest from
    device: str | None = None,  # TransformerLM's default
) -> TransformerLM:
    if cosine_cycle_iters is None:
        cosine_cycle_iters = max_iters
    if resume_from is not None:
        resume_from = resolve_checkpoint(Path(resume_from))
    if checkpoint_dir is None:
        # Resuming continues the original run's directory; a fresh run gets its own.
        checkpoint_dir = (
            resume_from.parent
            if resume_from is not None
            else CHECKPOINT_DIR / f"{Path(train).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
    # Model keys are logged separately below, since a resume overrides them from the checkpoint.
    print({k: v for k, v in locals().items() if k not in MODEL_KEYS})

    model_config = {k: v for k, v in locals().items() if k in MODEL_KEYS}
    if resume_from is not None and (saved := load_config(resume_from)) is not None:
        model_config = saved  # the architecture must match the weights we are about to load
        print(f"rebuilding from checkpoint config: {model_config}")

    train_data = open_dataset(train, model_config["vocab_size"])
    valid_data = open_dataset(valid, model_config["vocab_size"])

    model = TransformerLM(
        **model_config,
        device=torch.device(device) if device is not None else None,
    )
    # Read back what the submodules resolved, rather than recomputing it here, so the
    # config we persist holds concrete numbers instead of None.
    device = model.device
    context_length = model_config["context_length"]
    model_config["d_ff"] = model.layers[0].ffn.w1.weight.shape[0]

    # Only forward what was explicitly set, so AdamW's own defaults stay authoritative.
    adamw_kwargs = {"betas": tuple(betas)} if betas is not None else {}
    if eps is not None:
        adamw_kwargs["eps"] = eps
    optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay, **adamw_kwargs)
    # Materialize once: gradient_clipping iterates its argument twice, so a generator would
    # be exhausted after the norm pass and silently skip the rescale.
    params = list(model.parameters())
    print(f"{sum(p.numel() for p in params):,} parameters on {device}, config {model_config}")

    start_iter = 0
    if resume_from is not None:
        start_iter = load_checkpoint(resume_from, model, optimizer)
        print(f"resumed from {resume_from} at iteration {start_iter}")

    def checkpoint(it: int) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        out = checkpoint_dir / f"{it:07d}.pt"
        save_checkpoint(model, optimizer, it, out, model_config)
        print(f"iter {it:>7}  saved {out}")

    def valid_loss() -> float:
        return evaluate(model, valid_data, batch_size, context_length, device, eval_iters)

    model.train()
    t0 = time.time()
    for it in range(start_iter, max_iters):
        lr = cosine_learning_rate_schedule(it, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = load_data(train_data, batch_size, context_length, device)
        loss = compute_loss(model, x, y)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(params, max_grad_norm)
        optimizer.step()

        if log_every and it % log_every == 0:
            print(f"iter {it:>7}  loss {loss.item():.4f}  lr {lr:.2e}  {time.time() - t0:7.1f}s")
        if eval_every and it > 0 and it % eval_every == 0:
            print(f"iter {it:>7}  valid loss {valid_loss():.4f}")
        if checkpoint_every and it > 0 and it % checkpoint_every == 0:
            checkpoint(it)

    print(f"final valid loss {valid_loss():.4f}")
    checkpoint(max_iters)
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Every flag defaults to None; unset flags are dropped so train_llm's defaults win."""
    p = argparse.ArgumentParser(argument_default=None)
    p.add_argument("--train", type=Path, required=True, help=".npy of uint16 token IDs")
    p.add_argument("--valid", type=Path, required=True)

    p.add_argument("--vocab-size", type=int)
    p.add_argument("--context-length", type=int)
    p.add_argument("--d-model", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--d-ff", type=int)
    p.add_argument("--rope-theta", type=float)

    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-iters", type=int)
    p.add_argument("--lr-max", type=float)
    p.add_argument("--lr-min", type=float)
    p.add_argument("--warmup-iters", type=int)
    p.add_argument("--cosine-cycle-iters", type=int)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--betas", type=float, nargs=2)
    p.add_argument("--eps", type=float)
    p.add_argument("--max-grad-norm", type=float)

    p.add_argument("--log-every", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--eval-iters", type=int)
    p.add_argument("--checkpoint-every", type=int)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--device", type=str)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_llm(**{k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main()
