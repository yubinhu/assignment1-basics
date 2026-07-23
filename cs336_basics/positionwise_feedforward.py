import torch
from torch import nn
from jaxtyping import Bool, Float, Int
from einops import einsum
from torch import Tensor

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        if not d_ff:
            d_ff = int(round(8/3*d_model / 64) * 64)
        std = (2 / (d_ff + d_model))**0.5
        self.W1 = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(d_ff, d_model, dtype=dtype),
            mean = 0,
            std = std,
            a = -3 * std,
            b = 3 * std,
        ))
        self.W2 = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(d_model, d_ff, dtype=dtype),
            mean = 0,
            std = std,
            a = -3 * std,
            b = 3 * std,
        ))
        self.W3 = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(d_ff, d_model, dtype=dtype),
            mean = 0,
            std = std,
            a = -3 * std,
            b = 3 * std,
        ))
        

    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        W1X = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        SiLU = W1X * torch.sigmoid(W1X)
        W3X = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        SILUW3X = SiLU * W3X
        out = einsum(self.W2, SILUW3X, "d_model d_ff, ... d_ff -> ... d_model")
        return out