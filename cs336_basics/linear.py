import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        std = (2 / (in_features + out_features))**0.5
        self.W = nn.Parameter(nn.init.trunc_normal_(
            torch.empty(out_features, in_features, dtype=dtype),
            mean = 0, 
            std = std,
            a = -3 * std,
            b = 3 * std
            ))

        if not device:
            # Use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return out