import torch
from jaxtyping import Float
from torch import Tensor
from torch import nn

from cs336_basics.linear import Linear


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        if not d_ff:
            d_ff = int(round(8 / 3 * d_model / 64) * 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)
        silu_w3_x = silu * w3_x
        return self.w2(silu_w3_x)
