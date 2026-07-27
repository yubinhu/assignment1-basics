"""Regenerate every section 7 figure in the writeup from Weights and Biases history.

    uv run python -m cs336_basics.plot_curves

Writes into figures/: curves-lr.png (the 7.2 learning-rate sweep), curves-batchsize.png (the 7.2
batch-size sweep, fixed vs re-tuned LR), curves-ablations.png (the four 7.3 ablations against the
b128/lr 2e-3 baseline), and curves-owt.png (7.4, OWT vs TinyStories at the same budget). The
ablation runs come from `modal run --detach cs336_basics/train_modal.py::sweep --spec
cs336_basics/ablation_sweep.json` plus a follow-up nonorm probe at lr 1.5e-3.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

api = wandb.Api()
ENTITY = "harveyhu"
FIGURES = Path("figures")
TOKEN_BUDGET = 327_680_000


def num(v):
    """W&B history serializes non-finite floats as the strings 'NaN'/'Infinity'."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def runs_with_curves(proj):
    """Config plus the finite val-loss curve for every run in a project."""
    out = []
    for stub in api.runs(f"{ENTITY}/{proj}"):
        r = api.run(f"{ENTITY}/{proj}/{stub.id}")
        h = r.history(keys=["val_loss"], pandas=False)
        pts = sorted(
            (x["_step"], num(x["val_loss"]))
            for x in h
            if x.get("val_loss") is not None and math.isfinite(num(x["val_loss"]))
        )
        if pts:
            out.append(dict(name=r.name.rsplit("-2026", 1)[0], config=r.config, pts=pts))
    return out


def fig_lr():
    """All seven LR-sweep runs; color is peak LR, line style is warmup length."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {1e-3: "teal", 2e-3: "darkorange", 3e-3: "purple", 1e-2: "crimson"}
    # One curve per (lr, warmup): the project holds a duplicate 1e-3/200 run; keep the better.
    best = {}
    for d in runs_with_curves("tinystories_lr_sweep"):
        key = (d["config"]["lr_max"], d["config"]["warmup_iters"])
        if key not in best or min(v for _, v in d["pts"]) < min(v for _, v in best[key]["pts"]):
            best[key] = d
    for (lr, warm), d in sorted(best.items()):
        ax.plot(*zip(*d["pts"]), color=colors.get(lr, "0.5"),
                ls="-" if warm == 200 else "--", lw=1.6,
                label=f"lr {lr:.0e}, warmup {warm}")
    ax.axhline(1.45, color="0.5", lw=0.8, ls=":", label="1.45 target")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("validation loss")
    ax.set_ylim(1.3, 3.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Learning-rate sweep — batch 32 x 40,000 steps; divergence depends on warmup")
    fig.tight_layout()
    fig.savefig(FIGURES / "curves-lr.png", dpi=140)
    print("wrote figures/curves-lr.png")


def fig_batchsize():
    """Best run per batch size, and best-val vs batch size at fixed and re-tuned LR."""
    runs = []
    for d in runs_with_curves("tinystories_batchsize_sweep"):
        c = d["config"]
        # The b1 run is deliberately short and killed runs never finished; neither is a
        # batch-size result, so both stay out of the comparison.
        if c["total_tokens"] != TOKEN_BUDGET or d["pts"][-1][0] < c["max_iters"]:
            continue
        runs.append(dict(bs=c["batch_size"], lr=c["lr_max"], ctx=c["context_length"],
                         pts=d["pts"], best=min(v for _, v in d["pts"])))

    fixed = {d["bs"]: d for d in runs if d["lr"] == 1e-3}
    tuned = {}
    for d in runs:
        if d["lr"] != 1e-3 and (d["bs"] not in tuned or d["best"] < tuned[d["bs"]]["best"]):
            tuned[d["bs"]] = d
    for bs, d in fixed.items():  # batch sizes never re-tuned keep their 1e-3 run as best known
        tuned.setdefault(bs, d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    sizes = sorted(tuned)
    for i, bs in enumerate(sizes):
        d = tuned[bs]
        toks = [s * d["bs"] * d["ctx"] for s, _ in d["pts"]]
        axes[0].plot(toks, [v for _, v in d["pts"]], color=cmap(i / max(1, len(sizes) - 1)),
                     lw=1.7, label=f"bs={bs}, lr={d['lr']:.0e}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("tokens processed")
    axes[0].set_ylabel("validation loss")
    axes[0].set_ylim(1.3, 3.0)
    axes[0].set_title("Best run at each batch size (LR re-tuned)")
    axes[0].legend(fontsize=8)

    bs_f = sorted(fixed)
    axes[1].plot(bs_f, [fixed[b]["best"] for b in bs_f], "o-", color="crimson",
                 label="lr fixed at 1e-3")
    bs_t = sorted(tuned)
    axes[1].plot(bs_t, [tuned[b]["best"] for b in bs_t], "o-", color="teal", label="lr re-tuned")
    for b in bs_t:
        if tuned[b]["lr"] != 1e-3:
            axes[1].annotate(f"{tuned[b]['lr']:.0e}", (b, tuned[b]["best"]),
                             textcoords="offset points", xytext=(0, -14), ha="center",
                             fontsize=7, color="teal")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("best validation loss")
    axes[1].set_title("Re-tuning LR removes most of the large-batch penalty")
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.axhline(1.45, color="0.5", lw=0.8, ls=":")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "curves-batchsize.png", dpi=140)
    print("wrote figures/curves-batchsize.png")


def fig_ablations_and_owt():
    base = next(d["pts"] for d in runs_with_curves("tinystories_batchsize_sweep")
                if d["name"] == "b128lr2e-3")
    abl = {d["name"]: d["pts"] for d in runs_with_curves("tinystories_ablations")}
    owt = {d["name"]: d["pts"] for d in runs_with_curves("owt_main")}

    panels = [
        ("Ablation 1: remove RMSNorm",
         [("nonorm-lr2e-3", "crimson"), ("nonorm-lr1.5e-3", "purple"),
          ("nonorm-lr1e-3", "darkorange"), ("nonorm-lr3e-4", "teal")]),
        ("Ablation 1b: post-norm blocks", [("postnorm-lr2e-3", "crimson")]),
        ("Ablation 2: NoPE (no position embeddings)", [("nope-lr2e-3", "crimson")]),
        ("Ablation 3: SiLU FFN (no gate)", [("silu-lr2e-3", "crimson")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, series) in zip(axes.flat, panels):
        ax.plot(*zip(*base), color="0.15", lw=1.8, label="baseline (pre-norm SwiGLU RoPE, lr 2e-3)")
        for name, color in series:
            if name not in abl:
                print(f"missing from W&B: {name}")
                continue
            lr = name.rsplit("lr", 1)[-1]
            label = {"nonorm": f"no RMSNorm, lr {lr}", "postnorm": "post-norm, lr 2e-3",
                     "nope": "NoPE, lr 2e-3", "silu": "SiLU FFN, lr 2e-3"}[name.split("-")[0]]
            ax.plot(*zip(*abl[name]), color=color, lw=1.6, label=label)
        if title.startswith("Ablation 1:"):
            # The lr=2e-3 arm's only finite val point is 17,719 at step 500 (NaN from step
            # 1000); it cannot appear inside these axes, so say so instead of dropping it.
            ax.annotate("no RMSNorm @ lr 2e-3: val 1.8e4 at step 500,\n"
                        "NaN from step 1000 (off scale)",
                        xy=(500, 3.95), fontsize=8, color="crimson", va="top")
        ax.set_title(title)
        ax.set_xlabel("gradient step")
        ax.set_ylabel("validation loss")
        ax.set_ylim(1.3, 4.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Section 7.3 ablations — TinyStories, batch 128 x 10,000 steps (327.68M tokens)",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURES / "curves-ablations.png", dpi=140)
    print("wrote figures/curves-ablations.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, color in [("b128lr2e-3", "crimson"), ("b128lr1e-3", "darkorange")]:
        if name in owt:
            ax.plot(*zip(*owt[name]), color=color, lw=1.7,
                    label=f"OpenWebText, lr {name.rsplit('lr', 1)[-1]}")
    ax.plot(*zip(*base), color="0.15", lw=1.4, ls="--", label="TinyStories baseline (10k vocab)")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("validation loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_title("Section 7.4 — same model and step budget, OWT (32k vocab) vs TinyStories")
    fig.tight_layout()
    fig.savefig(FIGURES / "curves-owt.png", dpi=140)
    print("wrote figures/curves-owt.png")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    fig_lr()
    fig_batchsize()
    fig_ablations_and_owt()
