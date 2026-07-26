"""Run the section 7 training loop on Modal.

    modal secret create wandb-secret WANDB_API_KEY=<key>     # once
    modal run cs336_basics/train_modal.py --name baseline --max-iters 40000
    modal run cs336_basics/train_modal.py --project sweep --name lr-3e-4 --lr-max 3e-4

Everything the run produces (checkpoints, config.json, metrics.jsonl) lands on the
cs336-basics-data volume under data/checkpoints/<project>/<name>-<timestamp>/, so it survives the
container and can be pulled back down with `modal volume get`. Weights and Biases logs live
mirror through the wandb-secret; without that secret ExperimentLogger degrades to JSONL only
and training carries on.
"""

import modal

from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image

CHECKPOINTS = f"/root/{DATA_PATH}/checkpoints"


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="H100",
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
    max_iters: int = 40_000,
    batch_size: int = 32,
    lr_max: float = 1e-3,
    warmup_iters: int = 200,
    rope_theta: float = 10_000.0,
    resume_from: str | None = None,
):
    kwargs = {
        "project": project,
        "name": name,
        "max_iters": max_iters,
        "batch_size": batch_size,
        "lr_max": lr_max,
        "warmup_iters": warmup_iters,
        "rope_theta": rope_theta,
        "resume_from": resume_from,
    }
    run_dir = train_remote.remote(train, valid, **{k: v for k, v in kwargs.items() if v is not None})
    print(f"run directory on the volume: {run_dir}")
    print(f"fetch with: modal volume get cs336-basics-data {run_dir.split('/root/')[-1]}")
