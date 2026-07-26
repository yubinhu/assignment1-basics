import os
import typing

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    config: dict | None = None,
) -> None:
    """Dump everything needed to resume training: weights, optimizer state, and the step count.

    `config` records the kwargs the model was built with. Shapes alone are not enough to
    reconstruct it: num_heads is invisible (all attention matrices are d_model x d_model),
    and context_length / rope_theta live only in RoPE's non-persistent buffers.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore the model and optimizer in place, and return the iteration we left off at."""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def load_config(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> dict | None:
    """Read back the config saved by save_checkpoint, so a model can be rebuilt before loading."""
    return torch.load(src, map_location="cpu").get("config")
