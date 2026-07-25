from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores : Float[Tensor, " ... queries keys"] = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / d_k**0.5
    if mask is not None:
        mask = ~mask
        scores = scores.masked_fill(mask, float("-inf"))
    from cs336_basics.softmax import softmax
    weights = softmax(scores, dim=-1)
    out = einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return out