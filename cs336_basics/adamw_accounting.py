"""Resource accounting for training a Transformer LM with AdamW (problem adamw_accounting).

All tensors are assumed to be float32 (4 bytes). Notation follows the writeup:
    V = vocab_size, T = context_length, L = num_layers, d = d_model, h = num_heads,
    B = batch_size, and d_ff = 8/3 * d unless overridden.
"""

from dataclasses import dataclass

BYTES_PER_FLOAT = 4


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int | None = None  # defaults to 8/3 * d_model

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 8 * self.d_model / 3


GPT2_XL = ModelConfig(
    vocab_size=50_257,
    context_length=1_024,
    num_layers=48,
    d_model=1_600,
    num_heads=25,
)


# (a) Memory ------------------------------------------------------------------


def num_parameters(c: ModelConfig) -> float:
    """2Vd + L(4d^2 + 3d*d_ff + 2d) + d, which is 2Vd + L(12d^2 + 2d) + d at d_ff = 8/3 d.

    2Vd         input embedding (V x d) and untied output embedding (d x V)
    4d^2        per layer: W_Q, W_K, W_V, W_O, each d x d
    3d*d_ff     per layer: SwiGLU W_1, W_3 (d_ff x d) and W_2 (d x d_ff)
    2d          per layer: the two RMSNorm gain vectors
    d           the final RMSNorm gain vector
    """
    V, T, L, d, h, d_ff = _unpack(c)
    per_layer = 4 * d**2 + 3 * d * d_ff + 2 * d
    return 2 * V * d + L * per_layer + d


def num_activations(c: ModelConfig, batch_size: int) -> float:
    """B * [L * (56/3 * Td + 2hT^2) + Td + 2TV] at d_ff = 8/3 d.

    Per transformer block, per sequence:
        2Td         the two RMSNorm outputs
        3Td         Q, K, V projections
        hT^2        QK^T attention scores
        hT^2        softmax over the scores
        Td          attention-weighted sum of values
        Td          attention output projection
        4T*d_ff     SwiGLU: W_1 output, SiLU on the gate, W_3 output, elementwise product
        Td          SwiGLU W_2 output
    Then once at the end of the network, outside the L(...):
        Td          final RMSNorm
        TV          output embedding (logits)
        TV          cross entropy on the logits
    """
    V, T, L, d, h, d_ff = _unpack(c)
    per_layer = 8 * T * d + 2 * h * T**2 + 4 * T * d_ff
    return batch_size * (L * per_layer + T * d + 2 * T * V)


def memory_breakdown(c: ModelConfig, batch_size: int) -> dict[str, float]:
    """Peak training memory in bytes, decomposed. Gradients and the AdamW (m, v) state
    are each the same shape as the parameters, so the non-activation total is 4x params."""
    params = num_parameters(c)
    activations = num_activations(c, batch_size)
    return {
        "parameters": BYTES_PER_FLOAT * params,
        "gradients": BYTES_PER_FLOAT * params,
        "optimizer_state": BYTES_PER_FLOAT * 2 * params,  # first moment m and second moment v
        "activations": BYTES_PER_FLOAT * activations,
        "total": BYTES_PER_FLOAT * (4 * params + activations),
    }


# (b) Max batch size ----------------------------------------------------------


def memory_coefficients(c: ModelConfig) -> tuple[float, float]:
    """Return (a, b) in GB such that peak memory = a * batch_size + b."""
    a = BYTES_PER_FLOAT * num_activations(c, batch_size=1) / 1e9
    b = BYTES_PER_FLOAT * 4 * num_parameters(c) / 1e9
    return a, b


def max_batch_size(c: ModelConfig, budget_gb: float = 80.0) -> int:
    a, b = memory_coefficients(c)
    return int((budget_gb - b) // a)


# (c) FLOPs of one AdamW step -------------------------------------------------

# Elementwise FLOPs per parameter for algorithm 1, ignoring the O(1) computation of alpha_t:
#   theta <- theta - alpha*lambda*theta            2  (one multiply by the folded alpha*lambda, one subtract)
#   m     <- b1*m + (1-b1)*g                       3  (two multiplies, one add)
#   v     <- b2*v + (1-b2)*g^2                     4  (g^2, two multiplies, one add)
#   theta <- theta - alpha_t * m/(sqrt(v)+eps)     5  (sqrt, add eps, divide, multiply, subtract)
ADAMW_FLOPS_PER_PARAM = 14


def adamw_step_flops(c: ModelConfig) -> float:
    """FLOPs for the optimizer update itself (not the forward/backward that produced g)."""
    return ADAMW_FLOPS_PER_PARAM * num_parameters(c)


# (d) Training time -----------------------------------------------------------


def forward_flops(c: ModelConfig, batch_size: int) -> float:
    """Matrix-multiply FLOPs for one forward pass: B * [L(8Td^2 + 4T^2 d + 6Td*d_ff) + 2TdV]."""
    V, T, L, d, h, d_ff = _unpack(c)
    per_layer = (
        6 * T * d**2  # Q, K, V projections
        + 2 * T**2 * d  # QK^T
        + 2 * T**2 * d  # attention-weighted values
        + 2 * T * d**2  # attention output projection
        + 4 * T * d * d_ff  # SwiGLU W_1 and W_3
        + 2 * T * d * d_ff  # SwiGLU W_2
    )
    return batch_size * (L * per_layer + 2 * T * d * V)


def training_hours(
    c: ModelConfig,
    steps: int,
    batch_size: int,
    peak_flops_per_s: float = 495e12,
    mfu: float = 0.5,
) -> float:
    """Wall clock assuming the backward pass costs twice the forward pass."""
    flops_per_step = 3 * forward_flops(c, batch_size)
    return steps * flops_per_step / (peak_flops_per_s * mfu) / 3600


def _unpack(c: ModelConfig):
    return c.vocab_size, c.context_length, c.num_layers, c.d_model, c.num_heads, c.d_ff


if __name__ == "__main__":
    c = GPT2_XL
    print(f"GPT-2 XL: {c}\n")

    print("(a) peak memory, at batch_size = 1")
    for name, nbytes in memory_breakdown(c, batch_size=1).items():
        print(f"  {name:>16}: {nbytes / 1e9:10.3f} GB  ({nbytes / 2**30:8.3f} GiB)")
    print(f"  parameters: {num_parameters(c):,.0f}\n")

    a, b = memory_coefficients(c)
    print("(b) peak memory as a function of batch size")
    print(f"  {a:.3f} * batch_size + {b:.3f} GB")
    print(f"  max batch size in 80 GB: {max_batch_size(c)}\n")

    print("(c) FLOPs for one AdamW step")
    print(f"  {ADAMW_FLOPS_PER_PARAM} * num_parameters = {adamw_step_flops(c):.3e} FLOPs")
    print(f"  vs. one forward pass at batch_size 1: {forward_flops(c, 1):.3e} FLOPs\n")

    steps, batch_size = 400_000, 1_024
    hours = training_hours(c, steps=steps, batch_size=batch_size)
    print(f"(d) training {steps:,} steps at batch size {batch_size} on one H100 @ 50% MFU")
    print(f"  {3 * forward_flops(c, batch_size):.3e} FLOPs/step")
    print(f"  {hours:,.0f} hours ({hours / 24:,.0f} days)")
