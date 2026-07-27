"""Run the section 7 training loop on Modal.

    modal secret create wandb-secret WANDB_API_KEY=<key>     # once
    modal run cs336_basics/train_modal.py --name baseline
    modal run cs336_basics/train_modal.py --project sweep --name lr-3e-4 --lr 3e-4

Everything the run produces (checkpoints, config.json, metrics.jsonl) lands on the
cs336-basics-data volume under data/checkpoints/<project>/<name>-<timestamp>/, so it survives the
container and can be pulled back down with `modal volume get`. Weights and Biases logs live
mirror through the wandb-secret; without that secret ExperimentLogger degrades to JSONL only
and training carries on.
"""

import modal

from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image

CHECKPOINTS = f"/root/{DATA_PATH}/checkpoints"
# The GPU the handout quotes its 20-30 minute reference runtime against. Read at import, so it
# cannot be a CLI flag; edit here (H100, H200, A100-80GB, ...) to switch hardware.
GPU = "B200"


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu=GPU,
    timeout=12 * 60 * 60,
)
def train_remote(train: str, valid: str, **kwargs) -> str:
    """Returns the run directory rather than the model, which is not worth serializing back."""
    import os

    # Pins the run directory onto the volume instead of depending on the container's working
    # directory. Must precede the import, since CHECKPOINT_DIR is read at module scope.
    os.environ["CS336_CHECKPOINT_DIR"] = CHECKPOINTS

    import torch

    from cs336_basics.training_together import train_llm

    torch.set_float32_matmul_precision("high")  # TF32 matmuls; see the MFU discussion

    model = train_llm(DATA_PATH / train, DATA_PATH / valid, **kwargs)
    return str(model.run_dir)


@app.local_entrypoint()
def main(
    train: str = "tinystories_train_ids.npy",
    valid: str = "tinystories_valid_ids.npy",
    project: str | None = None,
    name: str | None = None,
    vocab_size: int | None = None,
    max_iters: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    warmup_iters: int | None = None,
    rope_theta: float | None = None,  # 0 disables position embeddings (NoPE)
    norm: str | None = None,  # "pre" | "post" | "none"
    ffn_type: str | None = None,  # "swiglu" | "silu"
    resume_from: str | None = None,
):
    """Everything defaults to None and unset flags are dropped, so train_llm owns the defaults.

    Restating them here once let this entrypoint drift to a different token budget than the local
    one; there is nothing to keep in sync now.
    """
    kwargs = {
        "project": project,
        "name": name,
        "vocab_size": vocab_size,
        "max_iters": max_iters,
        "batch_size": batch_size,
        "lr_max": lr,
        "warmup_iters": warmup_iters,
        "rope_theta": rope_theta,
        "norm": norm,
        "ffn_type": ffn_type,
        "resume_from": resume_from,
    }
    run_dir = train_remote.remote(train, valid, **{k: v for k, v in kwargs.items() if v is not None})
    print(f"run directory on the volume: {run_dir}")
    print(f"fetch with: modal volume get cs336-basics-data {run_dir.split('/root/')[-1]}")


@app.local_entrypoint()
def sweep(spec: str):
    """Launch every run in a JSON file concurrently under ONE app, one GPU each.

    Modal rate-limits app creation, so a sweep of `modal run` invocations needs minutes of
    spacing and retry loops; a single app spawning N calls does not. The spec is a JSON list
    of train_remote kwarg dicts, each needing at least {"train": ..., "valid": ...}. Run with
    --detach so the sweep survives this client going away.
    """
    import json
    from pathlib import Path

    runs = json.loads(Path(spec).read_text())
    calls = []
    for kwargs in runs:
        train, valid = kwargs.pop("train"), kwargs.pop("valid")
        calls.append((kwargs.get("name", "?"), train_remote.spawn(train, valid, **kwargs)))
        print(f"spawned {calls[-1][0]}")
    # Collect in launch order: a crash in one run (OOM, NaN assertions) must not orphan the rest.
    for name, call in calls:
        try:
            print(f"finished {name}: {call.get()}")
        except Exception as e:
            print(f"FAILED {name}: {e}")
