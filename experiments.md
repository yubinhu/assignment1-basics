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

A healthy run holds all three roughly flat. A divergent one (`--lr 3.0`) looks like:

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

At batch_size 32 and context_length 256 that is 40,000 steps, which is what `train_llm` defaults to,
so the whole table above is just its defaults. Every run prints and records `total_tokens`, so a
run that misses the budget is visible in `config.json` rather than only in the step count. Tuned by
trial and error below: learning rate, warmup, AdamW betas/eps, weight decay.

```bash
uv run python -m cs336_basics.training_together \
    --train data/tinystories_train_ids.npy --valid data/tinystories_valid_ids.npy \
    --eval-every 1000 --checkpoint-every 5000
```

## `learning_rate` — tune the learning rate

Target: validation loss <= 1.45 (or <= 2.00 on CPU/MPS with 40M tokens).

"Learning rate" here is α_max, the post-warmup peak — the quantity warmup ramps to, and the one
divergence is a property of. α_min is not swept independently: `lr_min` defaults to `lr_max / 10`
(the LLaMA/Chinchilla ratio) so each point in the sweep varies the schedule's height and not its
shape. Pinning `lr_min` at an absolute value would instead make the bottom of the sweep a constant
schedule, and anything below it an increasing one.

**Warmup is not handout-specified** — §7.2.1 lists it alongside learning rate as something to find
by trial and error. It turned out to matter as much as the learning rate itself, so the sweep is
two-dimensional: every learning rate was tried at `warmup_iters` of both 200 (0.5% of training) and
2000 (5%, the LLaMA value).

### Search strategy

Log-spaced probing outward from a 1e-3 anchor rather than a full grid. 1e-3 is the conventional
AdamW starting point at this scale, so it was run first, to completion, as a baseline. The second
probe went an order of magnitude up (1e-2) to bracket the instability boundary from above, on the
reasoning that the cheapest information early in a sweep is a *failure*: a diverging run identifies
itself within a few hundred steps and can be killed, costing a fraction of a full run. Bisecting
inward gave 3e-3, and when 3e-3 diverged at warmup 200 but converged at warmup 2000, warmup was
promoted from a fixed setting to a swept axis. 2e-3 then filled the remaining gap at warmup 200.

Divergent runs were killed as soon as the plateau was unambiguous rather than run to 40,000 steps,
which is why their step counts differ.

### Results

All runs: 40,000 steps at batch 32 x context 256 = 327,680,000 tokens, `lr_min = lr/10`.
"clip%" is the fraction of logged steps whose *pre-clip* gradient norm exceeded `max_grad_norm=1.0`.

| lr | warmup | Steps | Best val | Final val | clip% | max \|g\| | max act | \|w\| end | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1e-3 | 200 | 40,000 | **1.3799** | 1.4129 | 0% | 1.1 | 12.3 | 1856 | stable |
| 2e-3 | 200 | 40,000 | 1.3826 | 1.4038 | 0% | 1.2 | 28.8 | 1652 | stable |
| 3e-3 | 200 | 12,000 | 2.1360 | — | 46% | 15.8 | 798 | 2025 | **diverged** |
| 1e-2 | 200 | 19,500 | 2.4155 | — | 87% | 402.0 | 49,489 | 2663 | **diverged** |
| 1e-3 | 2000 | 40,000 | 1.3846 | 1.3890 | 0% | 2.3 | 12.8 | 1857 | stable |
| 3e-3 | 2000 | 40,000 | **1.3787** | 1.3787 | 0% | 1.1 | 29.4 | 1574 | stable (best) |
| 1e-2 | 2000 | 5,000 | 2.5328 | 2.9912 | 73% | 33.7 | 22,233 | 2930 | **diverged** |

(A 1e-3/warmup-200 run on a B200 was killed at 19,000 steps during hardware benchmarking; it tracked
the completed 5090 run to within 0.004 val loss and is omitted above.)

**Target met.** Four runs finish below the 1.45 requirement. The best is `lr=3e-3, warmup=2000` at
**1.3787**; `lr=1e-3, warmup=200` is second at **1.3799**.

### (a) Learning curves

![learning curves](data/curves-lr.png)

Left panel is the full range, where the three divergent runs sit in a band far above the rest.
Right panel zooms on the four converged runs — and the point of the zoom is that they are
essentially indistinguishable.

### (b) Edge of stability

**The divergence boundary is a function of warmup, not of learning rate alone.** This is the
sweep's main finding, and it is what makes a one-dimensional LR sweep misleading:

| warmup | Highest stable lr | Lowest divergent lr | Best lr found | Best lr / divergence point |
|---|---|---|---|---|
| 200 | 2e-3 | 3e-3 | 1e-3 (1.3799) | 3x below |
| 2000 | 3e-3 | 1e-2 | 3e-3 (1.3787) | 3.3x below |

A 10x increase in warmup bought roughly a 3x increase in usable learning rate. `lr=3e-3` is the
decisive case: at warmup 200 it plateaus at 2.1360 with clipping on 46% of steps, and at warmup 2000
it produces the best model in the sweep with clipping on 0% of steps. Identical peak learning rate,
opposite outcomes.

**Folk wisdom is not supported here.** The claim under test is that the best learning rate sits at
the edge of stability. It does not, in either warmup regime — and more strikingly, *how close you
get to the edge barely matters*. Across the four stable runs the peak learning rate varies 3x while
final validation loss varies from 1.3787 to 1.3846, a spread of 0.006. That is comparable to the
run-to-run noise: two runs of identical configuration on different hardware agreed to within 0.004,
and validation loss oscillates by roughly +/-0.03 between adjacent evaluations late in training.

So the practical conclusion is the opposite of "tune to the edge": within the stable band, the
learning rate is nearly a free parameter for final quality, and its real effect is on *risk*.
Running at 3e-3 requires getting warmup right or the run is destroyed; running at 1e-3 tolerates
either warmup. Given equal outcomes, the lower learning rate is the better engineering choice.

One measurable thing does change with learning rate: the final weight norm falls monotonically
(1856 -> 1652 -> 1574 as lr goes 1e-3 -> 2e-3 -> 3e-3), because AdamW's decoupled decay is scaled by
the learning rate. Higher learning rates reach a more heavily regularized solution of equal loss.

**Convergence rate.** The curves also do not separate on speed. All four stable runs cross 1.45 at
roughly 27,000-29,000 steps and are still improving slowly at 40,000. The higher learning rates buy
no wall-clock advantage at this scale — the cosine schedule spends most of training well below the
peak anyway, so the peak mostly determines whether the run survives its first few hundred steps.

### Divergence signature

The stable/divergent split is cleanly separable by the health monitors, with no overlap:

| | stable runs | divergent runs |
|---|---|---|
| clip% | 0-1% | 46-87% |
| max activation RMS | 12-30 | 798-49,489 |
| weight norm | decays (2238 -> 1574..1857) | grows (-> 2025..2930) |

**Divergence never produced a NaN.** Every divergent run kept finite losses and merely plateaued —
1e-2 at warmup 200 was still slowly descending at step 19,500, just from 3.2 toward 2.4 rather than
1.4. Gradient clipping is what prevents the blowup and in doing so conceals it: at 87% clipping the
effective step size is throttled by an arbitrary per-step factor, so the run limps rather than
explodes. Post-clip the gradient norm would read ~1.0 on every step and look healthy, which is why
`gradient_clipping` returns the norm measured *before* clipping.

Activation RMS is the earliest and highest-contrast signal. In the 1e-2 / warmup-200 run it fired
~300 steps before clipping did:

```
   100   lr 5.00e-03   |g| 0.53   |w| 2235.1   act     9.65   (mid-warmup, healthy)
   200   lr 1.00e-02   |g| 0.42   |w| 2239.0   act    74.65   (peak reached)
   300   lr 1.00e-02   |g| 0.41   |w| 2255.9   act   339.18   (|w| begins climbing)
   500   lr 1.00e-02   |g| 1.29   |w| 2304.3   act  1224.30   (clipping finally engages)
```

Note the trigger is *arrival at the peak*, not the ramp: at lr=5e-3 mid-warmup the run is clean, and
one step past the peak activation RMS is 53x its healthy baseline.

### Chosen configuration

`lr=1e-3, warmup=200`, validation loss **1.3799**. It ties the sweep's best within noise (0.0012
behind 3e-3/warmup-2000), sits a comfortable 3x below its divergence point, and is the least
sensitive to getting warmup right.

**Reproducibility caveats.** (1) This run used `warmup_iters=200` while the code default is now
2000, so reproducing it requires passing `--warmup-iters 200` explicitly. (2) There is no RNG seed,
so batch sampling differs between runs; single-minibatch `train_loss` differed by up to 0.27 between
two runs of identical configuration, though validation loss agreed to 0.004. Conclusions here rest
on validation loss for that reason.

**Reading configs back from W&B.** `wandb.Api().runs(project)` returns lazily-hydrated run objects
whose `.config` is an empty dict until `.load(force=True)` is called; `Api().run(path)` loads
eagerly and returns the full config. Analysis scripts that iterate `runs()` and read `.config`
directly will silently see no hyperparameters at all. Every `lr` and `warmup` in the table above was
cross-checked two independent ways — against the eagerly-loaded config, and against the logged `lr`
series (the step at which `lr` peaks is `warmup_iters`) — and all eight runs agree, at the exact
327,680,000-token budget.

## `batch_size_experiment` — batch size variations

Vary batch size from 1 to the memory limit, including 64 and 128. Re-tune LR where needed.

`max_iters` has to move inversely to keep the 327,680,000 token budget fixed, otherwise the
comparison confounds batch size with total compute:

| batch_size | `--max-iters` |
|---|---|
| 1 | 1,280,000 |
| 16 | 80,000 |
| 32 (base) | 40,000 |
| 64 | 20,000 |
| 128 | 10,000 |

| Run | batch_size | lr | Final val loss | Wall clock | Notes |
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
