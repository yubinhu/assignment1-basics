"""Regenerate the section 7.3/7.4 figures from Weights and Biases history.

    uv run python -m cs336_basics.plot_ablations

Writes data/curves-ablations.png (four panels, one per ablation, each against the batch-size
sweep's b128/lr 2e-3 baseline) and data/curves-owt.png (OWT at both learning rates with the
TinyStories baseline for scale). The runs themselves come from `modal run --detach
cs336_basics/train_modal.py::sweep --spec cs336_basics/ablation_sweep.json` plus a follow-up
nonorm probe at lr 1.5e-3.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

api = wandb.Api()
ENTITY = "harveyhu"


def num(v):
    """W&B history serializes non-finite floats as the strings 'NaN'/'Infinity'."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def curves(proj, want=None):
    """Finite val-loss curves per run, keyed by run name with the launch timestamp stripped."""
    out = {}
    for stub in api.runs(f"{ENTITY}/{proj}"):
        r = api.run(f"{ENTITY}/{proj}/{stub.id}")
        base = r.name.rsplit("-2026", 1)[0]
        if want is not None and base not in want:
            continue
        h = r.history(keys=["val_loss"], pandas=False)
        pts = sorted(
            (x["_step"], num(x["val_loss"]))
            for x in h
            if x.get("val_loss") is not None and math.isfinite(num(x["val_loss"]))
        )
        if pts:
            out[base] = pts
    return out


BASE = curves("tinystories_batchsize_sweep", want={"b128lr2e-3"})["b128lr2e-3"]
ABL = curves("tinystories_ablations")
OWT = curves("owt_main")

PANELS = [
    ("Ablation 1: remove RMSNorm",
     [("nonorm-lr2e-3", "crimson"), ("nonorm-lr1.5e-3", "purple"),
      ("nonorm-lr1e-3", "darkorange"), ("nonorm-lr3e-4", "teal")]),
    ("Ablation 1b: post-norm blocks", [("postnorm-lr2e-3", "crimson")]),
    ("Ablation 2: NoPE (no position embeddings)", [("nope-lr2e-3", "crimson")]),
    ("Ablation 3: SiLU FFN (no gate)", [("silu-lr2e-3", "crimson")]),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (title, series) in zip(axes.flat, PANELS):
    ax.plot(*zip(*BASE), color="0.15", lw=1.8, label="baseline (pre-norm SwiGLU RoPE, lr 2e-3)")
    for name, color in series:
        if name not in ABL:
            print(f"missing from W&B: {name}")
            continue
        lr = name.rsplit("lr", 1)[-1]
        label = {"nonorm": f"no RMSNorm, lr {lr}", "postnorm": "post-norm, lr 2e-3",
                 "nope": "NoPE, lr 2e-3", "silu": "SiLU FFN, lr 2e-3"}[name.split("-")[0]]
        ax.plot(*zip(*ABL[name]), color=color, lw=1.6, label=label)
    if title.startswith("Ablation 1:"):
        # The lr=2e-3 arm's only finite val point is 17,719 at step 500 (NaN from step 1000);
        # it cannot appear inside these axes, so say so instead of silently dropping it.
        ax.annotate("no RMSNorm @ lr 2e-3: val 1.8e4 at step 500,\nNaN from step 1000 (off scale)",
                    xy=(500, 3.95), fontsize=8, color="crimson", va="top")
    ax.set_title(title)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("validation loss")
    ax.set_ylim(1.3, 4.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Section 7.3 ablations — TinyStories, batch 128 x 10,000 steps (327.68M tokens)", y=0.995)
fig.tight_layout()
fig.savefig("data/curves-ablations.png", dpi=140)
print("wrote data/curves-ablations.png")

fig, ax = plt.subplots(figsize=(7.5, 5))
for name, color in [("b128lr2e-3", "crimson"), ("b128lr1e-3", "darkorange")]:
    if name in OWT:
        ax.plot(*zip(*OWT[name]), color=color, lw=1.7, label=f"OpenWebText, lr {name.rsplit('lr', 1)[-1]}")
ax.plot(*zip(*BASE), color="0.15", lw=1.4, ls="--", label="TinyStories baseline (10k vocab)")
ax.set_xlabel("gradient step")
ax.set_ylabel("validation loss")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
ax.set_title("Section 7.4 — same model and step budget, OWT (32k vocab) vs TinyStories")
fig.tight_layout()
fig.savefig("data/curves-owt.png", dpi=140)
print("wrote data/curves-owt.png")

for name, pts in sorted({**ABL, **OWT}.items()):
    print(f"{name:<22} best {min(v for _, v in pts):.4f}  final {pts[-1][1]:.4f}  last step {pts[-1][0]}")
