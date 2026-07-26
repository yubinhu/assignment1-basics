# Assigment 1 Write Up

### train_bpe_expts_owt
- (a) ÃÂ × 16. Makes sense but looks like OWT is a lower quality dataset than tiny stories. 

- (b) The longest tokens in tinystories are [(7160, b' accomplishment'), (9143, b' disappointment'), (9379, b' responsibility'), (3228, b' uncomfortable'), (3515, b' compassionate'), (5319, b' understanding'), (6386, b' neighbourhood'), (6497, b' Unfortunately'), (6874, b' determination'), (7756, b' encouragement')]
```bash
Top tokens in OWT: 
Token ID	Bytes	Decoded content
25822	64	ÃÂ × 16
25836	64	- × 64
31274	48	— × 16
10900	32	- × 32
15947	32	_ × 32
16885	32	ÃÂ × 8
25146	32	= × 32
28585	32	. × 32
31162	32	* × 32
15279	24	— × 8
```

The longest ones in tiny stories are more emotionally coded and the ones in OWT come from artifacts

### tokenizer_experiments
```bash
=== (a) Compression ratios on 10 sampled docs ===
TinyStories tokenizer on TinyStories sample: 4.187 bytes/token
OpenWebText tokenizer on OpenWebText sample: 4.702 bytes/token
  (#bytes ts=7533, #toks=1799; #bytes owt=31604, #toks=6721)

=== (b) Cross-tokenize: OWT sample with TinyStories tokenizer ===
OWT tok on OWT: 4.702 bytes/token (6721 toks)
TS  tok on OWT: 3.198 bytes/token (9882 toks)
Expect worse (higher bytes/token) with TS: less domain match ⇒ shorter merges / more tokens.
Sample decode prefix (TS tok): "What wouldn't you do to save someone you love?\n\nWhen They Come Calling is a modern ghost story, a suspenseful weaving of urban battles,"

=== (c) Throughput estimate ===
Encoded 10,003,969 bytes → 2,430,529 tokens in 0.94s
Throughput: 10,612,098 bytes/s (10.61 MB/s)
Time for The Pile (825GB): 23.2 hours (1.0 days)

=== (d) Serialize train/valid as uint16 ===
uint16 is appropriate because max token id < 65536 for both vocabs (10k / 32k), so 2 bytes/token is enough without wasting space like int32.
Encoding data/TinyStoriesV2-GPT4-train.txt → data/tinystories_train_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 541,229,347 tokens, max_id=9999 in 84.0s (26.5 MB/s)
Encoding data/TinyStoriesV2-GPT4-valid.txt → data/tinystories_valid_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 5,465,883 tokens, max_id=9999 in 0.9s (24.3 MB/s)
Encoding data/owt_train.txt → data/owt_train_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 2,727,120,452 tokens, max_id=31999 in 508.3s (23.5 MB/s)
Encoding data/owt_valid.txt → data/owt_valid_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 66,401,098 tokens, max_id=31999 in 13.6s (21.3 MB/s)
```

### transformer_accounting

#### (a)

For this bias-free architecture with untied input embeddings and LM head, the parameter count is
$2Vd + L(4d^2 + 3d d_{\mathrm{ff}} + 2d) + d = 1{,}640{,}452{,}800$. In single precision, the parameters
require $6{,}561{,}811{,}200$ bytes, or about $6.562$ GB ($6.111$ GiB).

#### (b)

Let $T=1{,}024$, $d=1{,}600$, $d_{\mathrm{ff}}=4{,}288$, $V=50{,}257$, $L=48$,
$h=25$, and $d_h=d/h=64$. The required matrix multiplies are:

| Matrix multiply | Multiplicity and shape | FLOPs |
|---|---:|---:|
| Query, key, and value projections | $3L$ multiplies of $(T \times d)(d \times d)$ | $6LTd^2 = 754{,}974{,}720{,}000$ |
| Attention scores $QK^\mathsf{T}$ | $Lh$ multiplies of $(T \times d_h)(d_h \times T)$ | $2LT^2d = 161{,}061{,}273{,}600$ |
| Attention-weighted values $AV$ | $Lh$ multiplies of $(T \times T)(T \times d_h)$ | $2LT^2d = 161{,}061{,}273{,}600$ |
| Attention output projection | $L$ multiplies of $(T \times d)(d \times d)$ | $2LTd^2 = 251{,}658{,}240{,}000$ |
| SwiGLU $W_1$ and $W_3$ projections | $2L$ multiplies of $(T \times d)(d \times d_{\mathrm{ff}})$ | $4LTd d_{\mathrm{ff}} = 1{,}348{,}888{,}166{,}400$ |
| SwiGLU $W_2$ projection | $L$ multiplies of $(T \times d_{\mathrm{ff}})(d_{\mathrm{ff}} \times d)$ | $2LTd d_{\mathrm{ff}} = 674{,}444{,}083{,}200$ |
| Final LM head | One multiply of $(T \times d)(d \times V)$ | $2TdV = 164{,}682{,}137{,}600$ |

Thus, the matrix multiplies require
$$
L(8Td^2 + 4T^2d + 6Td d_{\mathrm{ff}}) + 2TdV
= 3{,}516{,}769{,}894{,}400
$$
FLOPs, or about $3.517$ TFLOPs. The embedding lookup, RMSNorm, RoPE, softmax, activations, and residual
operations are not dense matrix multiplies and are excluded from this accounting.

#### (c)

The SwiGLU feed-forward layers dominate at $57.53\%$ of the FLOPs, while multi-head attention as a
whole uses $37.78\%$. Most of that attention cost is in the combined QKV and output projections
($28.62\%$); at this context length, the two quadratic attention products contribute only $9.16\%$.

#### (d)

Using $T=1{,}024$, $V=50{,}257$, and the nearest multiple of 64 to
$\frac{8}{3}d$ for $d_{\mathrm{ff}}$ gives $d_{\mathrm{ff}}=2{,}048$, $2{,}752$, $3{,}392$, and
$4{,}288$ for small, medium, large, and XL, respectively. Each component below is reported as
TFLOPs followed by its proportion of the model's total FLOPs.

| Model | QKV projections | $QK^\mathsf{T}$ | $AV$ | Output projection | SwiGLU FFN | LM head | Total TFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 0.0435 (14.91%) | 0.0193 (6.63%) | 0.0193 (6.63%) | 0.0145 (4.97%) | 0.1160 (39.76%) | 0.0790 (27.10%) | 0.2916 |
| Medium | 0.1546 (18.62%) | 0.0515 (6.21%) | 0.0515 (6.21%) | 0.0515 (6.21%) | 0.4155 (50.05%) | 0.1054 (12.70%) | 0.8302 |
| Large | 0.3624 (20.49%) | 0.0966 (5.46%) | 0.0966 (5.46%) | 0.1208 (6.83%) | 0.9603 (54.30%) | 0.1317 (7.45%) | 1.7685 |
| XL | 0.7550 (21.47%) | 0.1611 (4.58%) | 0.1611 (4.58%) | 0.2517 (7.16%) | 2.0233 (57.53%) | 0.1647 (4.68%) | 3.5168 |

As model size grows at fixed $T$ and $V$, the FFN share rises from $39.76\%$ to $57.53\%$, and the
combined projection share rises from $19.88\%$ to $28.62\%$, because these block operations scale
roughly as $Ld^2$. The quadratic attention-products share falls from $13.25\%$ to $9.16\%$, while the
LM-head share falls from $27.10\%$ to $4.68\%$.

#### (e)

At $T=16{,}384$, the XL forward pass costs $133{,}577{,}729{,}638{,}400$ FLOPs
($133.578$ TFLOPs), which is $37.983\times$ the cost at $T=1{,}024$. The quadratic attention
products grow from $9.16\%$ to $61.73\%$ of the total, while the combined QKV/output-projection, FFN,
and LM-head shares fall to $12.06\%$, $24.24\%$, and $1.97\%$, respectively.

### learning_rate_tuning
1 slowly dropped loss from 21 to 11. 1e1 dropped it slowly from 20 to 0.01. 1e2 quickly dropped it to a very small number then 0 afterwards. 1e3 diverged and went to inf after a bit. 

### adamw_accounting
GPT-2 XL: ModelConfig(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4266.666666666667)

(a) peak memory, at batch_size = 1
        parameters:      6.542 GB  (   6.093 GiB)
         gradients:      6.542 GB  (   6.093 GiB)
   optimizer_state:     13.084 GB  (  12.186 GiB)
       activations:     16.357 GB  (  15.233 GiB)
             total:     42.525 GB  (  39.605 GiB)
  parameters: 1,635,537,600

(b) peak memory as a function of batch size
  16.357 * batch_size + 26.169 GB
  max batch size in 80 GB: 3

(c) FLOPs for one AdamW step
  14 * num_parameters = 2.290e+10 FLOPs
  vs. one forward pass at batch_size 1: 3.507e+12 FLOPs

(d) training 400,000 steps at batch size 1024 on one H100 @ 50% MFU
  1.077e+16 FLOPs/step
  4,836 hours (202 days)

### experiment_log

Logging infrastructure is in [cs336_basics/experiment_log.py](cs336_basics/experiment_log.py); the
experiment log itself is [experiments.md](experiments.md).