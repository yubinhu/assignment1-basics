"""Experiment tracking for the section 7 runs (problem experiment_log).

Every run writes two files into its checkpoint directory:

    config.json     the resolved hyperparameters, so a run is reproducible from disk alone
    metrics.jsonl   one JSON object per logged event

Each metrics record carries both `step` and `wall_time` (seconds since training started), so
loss curves can be plotted against gradient steps or wall clock.

Runs also mirror to Weights and Biases by default, and the directory layout mirrors the W&B
hierarchy: `data/checkpoints/<project>/<name>/`. --project defaults to the dataset stem, --name is
the run name with a timestamp appended. Both are read back off the path here, so nothing needs
restating. Set WANDB_MODE=disabled to opt out; if wandb is unreachable the run keeps going with
JSONL only.

    uv run python -m cs336_basics.experiment_log table            # compare all runs
    uv run python -m cs336_basics.experiment_log plot -o curves.png
"""

import argparse
import json
import os
import time
from pathlib import Path

# Absolute on Modal, where the volume mounts at /root/data and cwd is not worth relying on.
CHECKPOINT_DIR = Path(os.environ.get("CS336_CHECKPOINT_DIR", "data/checkpoints"))
CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"


def project_name(run_dir: Path) -> str:
    """Run directories are laid out as <project>/<run name>, mirroring the W&B hierarchy."""
    return run_dir.parent.name


class ExperimentLogger:
    """Append-only JSONL writer, optionally mirroring to Weights and Biases."""

    def __init__(self, run_dir: Path, config: dict):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.t0 = time.time()

        serializable = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        (run_dir / CONFIG_FILE).write_text(json.dumps(serializable, indent=2))
        self.metrics = (run_dir / METRICS_FILE).open("a")

        # On by default. Set WANDB_MODE=disabled (or offline) to opt out; a failure here
        # degrades to JSONL-only rather than taking the training run down with it.
        self.wandb = None
        try:
            import wandb

            wandb.init(project=project_name(run_dir), name=run_dir.name, config=serializable)
            self.wandb = wandb
        except Exception as e:
            print(f"wandb disabled ({type(e).__name__}: {e}); still logging to {run_dir / METRICS_FILE}")

    def log(self, step: int, **metrics) -> None:
        record = {"step": step, "wall_time": round(time.time() - self.t0, 3), **metrics}
        self.metrics.write(json.dumps(record) + "\n")
        self.metrics.flush()  # so a killed run keeps everything up to the last logged step
        if self.wandb is not None:
            # `step` is wandb's own x-axis, so passing it through would add a redundant chart.
            self.wandb.log({k: v for k, v in record.items() if k != "step"}, step=step)

    def close(self) -> None:
        self.metrics.close()
        if self.wandb is not None:
            self.wandb.finish()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """Return (config, records) for one run directory."""
    config_path, metrics_path = run_dir / CONFIG_FILE, run_dir / METRICS_FILE
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    records = []
    if metrics_path.exists():
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    return config, records


def find_runs(root: Path) -> list[Path]:
    """Recursive, so both <root>/<project>/<name>/ and a bare <root>/<name>/ are picked up."""
    return sorted(p.parent for p in root.glob(f"**/{METRICS_FILE}"))


def table(runs: list[Path], keys: tuple[str, ...]) -> str:
    """Markdown table of one row per run, for pasting straight into the experiment log."""
    header = ["project", "run", *keys, "steps", "min val loss", "final val loss", "wall clock"]
    rows = [header, ["---"] * len(header)]
    for run_dir in runs:
        config, records = read_run(run_dir)
        val = [r for r in records if "val_loss" in r]
        rows.append(
            [
                project_name(run_dir),
                run_dir.name,
                *(str(config.get(k, "")) for k in keys),
                str(max((r["step"] for r in records), default=0)),
                f"{min((r['val_loss'] for r in val), default=float('nan')):.4f}",
                f"{val[-1]['val_loss']:.4f}" if val else "",
                f"{max((r['wall_time'] for r in records), default=0) / 60:.1f} min",
            ]
        )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def plot(runs: list[Path], out: Path) -> None:
    """Two panels of validation loss: against gradient steps, and against wall clock."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for run_dir in runs:
        _, records = read_run(run_dir)
        val = [r for r in records if "val_loss" in r]
        if not val:
            continue
        losses = [r["val_loss"] for r in val]
        axes[0].plot([r["step"] for r in val], losses, label=run_dir.name)
        axes[1].plot([r["wall_time"] / 60 for r in val], losses, label=run_dir.name)

    for ax, xlabel in zip(axes, ("gradient steps", "wall clock (minutes)")):
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation loss")
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("table", "plot"))
    p.add_argument("--runs", type=Path, nargs="*", help="run directories (default: all under --root)")
    p.add_argument("--root", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--keys", nargs="*", default=["lr_max", "batch_size", "d_model", "num_layers"])
    p.add_argument("-o", "--out", type=Path, default=Path("data/curves.png"))
    args = p.parse_args(argv)

    runs = args.runs or find_runs(args.root)
    if not runs:
        raise SystemExit(f"no runs with {METRICS_FILE} found under {args.root}")

    if args.command == "table":
        print(table(runs, tuple(args.keys)))
    else:
        plot(runs, args.out)


if __name__ == "__main__":
    main()
