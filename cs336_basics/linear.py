import torch
from einops import einsum
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
