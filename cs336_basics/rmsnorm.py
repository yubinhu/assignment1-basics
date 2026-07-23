import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.eps = eps
        if not device:
            # Use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gain.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intype = x.dtype
        d_model = x.shape[-1]
        assert x.shape[-1] == self.gain.shape[0], "[RMSNorm.forward] input and gain dimensions must match"
        x = x.to(torch.float32)
        RMS = ((1 / d_model) * torch.sum(x * x, dim=-1, keepdim=True) + self.eps)**0.5
        out = x / RMS * self.gain
        return out.to(intype)