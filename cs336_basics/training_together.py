"""Train a Transformer LM. Example:

    uv run python -m cs336_basics.training_together \
        --train data/tinystories_train_ids.npy --valid data/tinystories_valid_ids.npy \
        --lr 1e-3

Defaults are the section 7.2.1 TinyStories configuration, sized so batch_size x max_iters x
context_length is the handout's 327,680,000 token budget. --lr is the peak (post-warmup) learning
rate, which is the one the handout asks you to tune; --lr-min trails it at a tenth unless set.

Checkpoints land in data/checkpoints/<project>/<name>/<iteration>.pt, mirroring the Weights and
Biases hierarchy: --project defaults to the train file's stem, and --name is the run name with a
timestamp appended. To pick a run back up, point --resume-from at that directory (latest iteration
wins) or at a specific .pt file; the model config travels with the checkpoint, so no model flags
need repeating.

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
from cs336_basics.experiment_log import CHECKPOINT_DIR, ExperimentLogger
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import cosine_learning_rate_schedule
from cs336_basics.transformer_lm import TransformerLM

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


class ActivationMonitor:
    """RMS of the residual stream after the last block, captured only on the steps we log.

    Hooks ln_final and reads its *input*, not its output: RMSNorm rescales to unit RMS, so its
    output is pinned near 1 and cannot show explosion. The residual stream feeding it is what
    grows or collapses with depth. One probe is enough signal; hooking every block would cost
    more for little gain.
    """

    def __init__(self, model: TransformerLM):
        self.value = float("nan")
        self.enabled = False
        model.ln_final.register_forward_hook(self._hook, with_kwargs=False)

    def _hook(self, module, args, output) -> None:
        if self.enabled:
            self.value = args[0].detach().float().pow(2).mean().sqrt().item()
            self.enabled = False


def global_norm(tensors: list[torch.Tensor]) -> float:
    """L2 norm over all tensors concatenated, matching how gradient_clipping measures gradients."""
    return torch.sqrt(sum((t.detach().float() ** 2).sum() for t in tensors)).item()


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
    max_iters: int = 40_000,  # 327,680,000 tokens at batch_size 32, context_length 256
    lr_max: float = 1e-3,  # the handout's "learning rate": the post-warmup peak, exposed as --lr
    lr_min: float | None = None,  # defaults to lr_max / 10, the LLaMA/Chinchilla ratio
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
    project: str | None = None,  # W&B project; defaults to the train file's stem
    name: str | None = None,  # W&B run name; the timestamp is appended to keep it unique
    checkpoint_dir: Path | None = None,  # defaults to data/checkpoints/<project>/<name>/
    resume_from: Path | None = None,  # a checkpoint file, or a run directory to take the latest from
    device: str | None = None,  # TransformerLM's default
) -> TransformerLM:
    torch.set_float32_matmul_precision('high')
    if cosine_cycle_iters is None:
        cosine_cycle_iters = max_iters
    if lr_min is None:
        # Tied to the peak rather than absolute, so sweeping lr_max varies only its height and not
        # the schedule's shape. A fixed lr_min would make the low end of a sweep a constant
        # schedule, and anything under it an *increasing* one.
        lr_min = lr_max / 10
    if resume_from is not None:
        resume_from = resolve_checkpoint(Path(resume_from))
    # The directory mirrors the W&B hierarchy: <project>/<run name>. ExperimentLogger reads both
    # back off the path, so these names are stated once.
    project = project or Path(train).stem
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{name}-{stamp}" if name else stamp
    if checkpoint_dir is None:
        # Resuming continues the original run's directory; a fresh run gets its own.
        checkpoint_dir = (
            resume_from.parent if resume_from is not None else CHECKPOINT_DIR / project / name
        )
    run_config = dict(locals())
    # Model keys are logged separately below, since a resume overrides them from the checkpoint.
    print({k: v for k, v in run_config.items() if k not in MODEL_KEYS})

    model_config = {k: v for k, v in run_config.items() if k in MODEL_KEYS}
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
    num_params = sum(p.numel() for p in params)
    # Logged because section 7 holds it fixed while varying batch size, so it is the quantity
    # that has to match across runs rather than the step count.
    total_tokens = batch_size * max_iters * context_length
    print(f"{num_params:,} parameters on {device}, {total_tokens:,} tokens, config {model_config}")
    run_config.update(
        model_config, device=str(device), num_parameters=num_params, total_tokens=total_tokens
    )

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
    activations = ActivationMonitor(model)
    with ExperimentLogger(checkpoint_dir, run_config) as logger:
        for it in range(start_iter, max_iters):
            lr = cosine_learning_rate_schedule(it, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr

            logging_now = bool(log_every) and it % log_every == 0
            activations.enabled = logging_now

            x, y = load_data(train_data, batch_size, context_length, device)
            loss = compute_loss(model, x, y)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = gradient_clipping(params, max_grad_norm)  # measured before clipping
            optimizer.step()

            if logging_now:
                weight_norm, act_rms = global_norm(params), activations.value
                logger.log(
                    it,
                    train_loss=loss.item(),
                    lr=lr,
                    grad_norm=grad_norm,
                    weight_norm=weight_norm,
                    activation_rms=act_rms,
                )
                elapsed = time.time() - logger.t0
                print(
                    f"iter {it:>7}  loss {loss.item():.4f}  lr {lr:.2e}  "
                    f"|g| {grad_norm:8.3f}  |w| {weight_norm:8.1f}  "
                    f"act {act_rms:6.3f}  {elapsed:7.1f}s"
                )
            if eval_every and it > 0 and it % eval_every == 0:
                logger.log(it, val_loss=(val := valid_loss()))
                print(f"iter {it:>7}  valid loss {val:.4f}")
            if checkpoint_every and it > 0 and it % checkpoint_every == 0:
                checkpoint(it)

        logger.log(max_iters, val_loss=(val := valid_loss()))
        print(f"final valid loss {val:.4f}")
        checkpoint(max_iters)

    model.run_dir = checkpoint_dir  # so callers can find the artifacts without recomputing the name
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
    # The handout tunes a single "learning rate", which is the peak. lr_min rides along at
    # lr_max / 10 unless overridden, so --lr is the only knob a sweep needs.
    p.add_argument("--lr", dest="lr_max", type=float)
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
    p.add_argument("--project", type=str, help="W&B project; defaults to the train file's stem")
    p.add_argument("--name", type=str, help="W&B run name; a timestamp is appended")
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--device", type=str)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_llm(**{k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main()
