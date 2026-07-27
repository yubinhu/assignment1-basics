import torch
from jaxtyping import Float
from torch import Tensor
from torch import nn

from cs336_basics.linear import Linear


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        ffn_type: str = "swiglu",  # "silu" drops the gate (no w3), the section 7.3 ablation
    ):
        super().__init__()
        if ffn_type not in ("swiglu", "silu"):
            raise ValueError(f"ffn_type must be 'swiglu' or 'silu', got {ffn_type!r}")
        self.ffn_type = ffn_type
        if not d_ff:
            # SwiGLU has three weight matrices to SiLU's two, so the handout narrows it to
            # ~8/3 d_model (rounded for tensor cores) and widens SiLU to 4 d_model, keeping
            # the two ablation arms at approximately equal parameter counts.
            d_ff = int(round(8 / 3 * d_model / 64) * 64) if ffn_type == "swiglu" else 4 * d_model

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        if ffn_type == "swiglu":
            self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        if self.ffn_type == "silu":
            return self.w2(silu)
        return self.w2(silu * self.w3(x))
