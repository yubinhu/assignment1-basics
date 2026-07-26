# Experiment Log

Record of everything tried for section 7. Deliverable for problem `experiment_log`.

## Infrastructure

Every training run writes a self-contained directory under `data/checkpoints/`:

```
data/checkpoints/<project>/<name>-<YYYYmmdd-HHMMSS>/
    config.json      resolved hyperparameters (including num_parameters and resolved d_ff)
    metrics.jsonl    one record per event, each stamped with both `step` and `wall_time`
    0001000.pt       checkpoints, zero-padded so lexical order is numeric order
```

The layout mirrors the W&B hierarchy. `--project` groups runs and defaults to the train file's
stem; `--name` names the individual run and gets a timestamp appended so repeats never collide.

| Flags | Directory | W&B project | W&B run |
|---|---|---|---|
| _(none)_ | `tinystories_train_ids/20260726-050004` | `tinystories_train_ids` | `20260726-050004` |
| `--name lr-3e-4` | `tinystories_train_ids/lr-3e-4-20260726-050004` | `tinystories_train_ids` | `lr-3e-4-20260726-050004` |
| `--project ablations --name nope` | `ablations/nope-20260726-050004` | `ablations` | `nope-20260726-050004` |

`metrics.jsonl` records look like:

```json
{"step": 500, "wall_time": 61.4, "train_loss": 2.9104, "lr": 0.00089,
 "grad_norm": 0.412, "weight_norm": 1183.6, "activation_rms": 14.72}
{"step": 500, "wall_time": 61.7, "val_loss": 2.8733}
```

### Health monitoring

Per §7.2.3 ("monitor the norms of activations, model weights, and gradients"), every train record
carries three norms, echoed to the console as `|g|`, `|w|`, and `act`:

- **`grad_norm`** — global L2 over all gradients, measured *before* clipping. Post-clip it would
  be pinned at `max_grad_norm` and carry no information.
- **`weight_norm`** — global L2 over all parameters.
- **`activation_rms`** — RMS of the residual stream entering the final RMSNorm. Deliberately the
  norm's *input*: RMSNorm rescales its output to unit RMS, so the output cannot reveal explosion.

A healthy run holds all three roughly flat. A divergent one (`--lr-max 3.0`) looks like:

```
iter       0  loss  5.0212  |g|  1.581  |w|    94.9  act 1.659
iter      10  loss 262.079  |g| 39.662  |w|  4040.0  act 456721408.000
```

Note that `activation_rms` moves by eight orders of magnitude while `grad_norm` moves by one —
clipping bounds the gradient, so the activation norm is by far the most sensitive early warning.

Both axes required by the problem statement are therefore available for every point.

Set `WANDB_MODE=disabled` to opt out of W&B. The JSONL is written either way, so the `table`/`plot`
commands below never depend on the external service.

Comparing runs:

```bash
uv run python -m cs336_basics.experiment_log table                       # markdown summary
uv run python -m cs336_basics.experiment_log table --keys lr_max warmup_iters
uv run python -m cs336_basics.experiment_log plot -o data/curves.png     # val loss vs step and vs wall clock
uv run python -m cs336_basics.experiment_log plot --runs data/checkpoints/tinystories_train_ids-2026*
```

Resuming needs no model flags — the architecture travels inside the checkpoint:

```bash
uv run python -m cs336_basics.training_together \
    --train data/tinystories_train_ids.npy --valid data/tinystories_valid_ids.npy \
    --resume-from data/checkpoints/tinystories_train_ids-20260725-193103
```

## Base configuration

Per §7.2.1, held fixed unless a row below says otherwise.

| Hyperparameter | Value | Source |
|---|---|---|
| vocab_size | 10,000 | handout |
| context_length | 256 | handout |
| d_model | 512 | handout |
| d_ff | 1344 | handout; also what `PositionwiseFeedForward` derives from `d_model` |
| num_layers | 4 | handout |
| num_heads | 16 | handout |
| rope_theta | 10,000 | handout |
| total tokens | 327,680,000 | handout (batch_size x max_iters x context_length) |

At batch_size 32 and context_length 256 that is 40,000 steps. Tuned by trial and error below:
learning rate, warmup, AdamW betas/eps, weight decay.

```bash
uv run python -m cs336_basics.training_together \
    --train data/tinystories_train_ids.npy --valid data/tinystories_valid_ids.npy \
    --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 \
    --batch-size 32 --max-iters 40000 --eval-every 1000 --checkpoint-every 5000
```

## `learning_rate` — tune the learning rate

Target: validation loss <= 1.45 (or <= 2.00 on CPU/MPS with 40M tokens).

Search strategy: _(describe — e.g. log-spaced sweep, then bisect toward the divergence edge)_

| Run | lr_max | warmup | Final val loss | Notes |
|---|---|---|---|---|
| | 1e-4 | | | |
| | 3e-4 | | | |
| | 1e-3 | | | |
| | 3e-3 | | | |
| | 1e-2 | | | |

_(a) Learning curves:_ `data/curves-lr.png`

_(b) Edge of stability._ Lowest diverging LR: ___. Best LR: ___. Ratio: ___.
Discussion: _(how close is the best LR to the divergence boundary, and what that implies)_

## `batch_size_experiment` — batch size variations

Vary batch size from 1 to the memory limit, including 64 and 128. Re-tune LR where needed.

| Run | batch_size | lr_max | Final val loss | Wall clock | Notes |
|---|---|---|---|---|---|
| | 1 | | | | |
| | 16 | | | | |
| | 64 | | | | |
| | 128 | | | | |
| | max | | | | |

Findings: _(a few sentences on the throughput/quality tradeoff)_

## `generate` — sample text

Checkpoint used: ___. Decoding parameters: temperature ___, top-p ___.

```
(paste >= 256 tokens of generated text)
```

Fluency comment and two factors affecting quality: _(fill in)_

## Ablations

Each is a 0.5 B200 hr run against the tuned base config.

| Problem | Change | Final val loss | Delta vs base | Notes |
|---|---|---|---|---|
| `layer_norm_ablation` | remove RMSNorm | | | |
| `pre_norm_ablation` | post-norm instead of pre-norm | | | |
| `no_pos_emb` | NoPE (no RoPE) | | | |
| `swiglu_ablation` | SiLU FFN instead of SwiGLU | | | |

## `main_experiment` — OpenWebText

| Run | Change vs TinyStories base | Final val loss | Notes |
|---|---|---|---|

Comparison of OWT vs TinyStories losses and generated text: _(fill in)_

## `leaderboard`

| Run | Change | Val loss | Wall clock | Notes |
|---|---|---|---|---|

Best submitted configuration: _(fill in)_
