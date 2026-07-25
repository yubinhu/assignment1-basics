import torch
from torch import nn, Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Int

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    batch_size = inputs.shape[0]
    c = torch.max(inputs, dim=-1, keepdim=True).values
    inputs = inputs - c
    denom = einsum(torch.exp(inputs), "batch_size vocab_size -> batch_size")
    # targets = rearrange(targets, "batch_size -> batch_size 1")
    batch_idx = torch.arange(inputs.shape[0], device=inputs.device)
    targets_logits = inputs[batch_idx, targets]
    l = 1 / batch_size * einsum( torch.log(denom) - targets_logits, "batch_size ->" )
    return l
