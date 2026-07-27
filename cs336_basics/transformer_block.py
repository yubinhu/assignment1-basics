import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = None,
        theta: float = None,
        device: torch.device = None,
        norm: str = "pre",  # "pre" | "post" | "none", the section 7.3 layer-norm ablations
        ffn_type: str = "swiglu",  # "swiglu" | "silu", the section 7.3 gating ablation
    ):
        super().__init__()
        if norm not in ("pre", "post", "none"):
            raise ValueError(f"norm must be 'pre', 'post', or 'none', got {norm!r}")
        self.norm = norm
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, ffn_type=ffn_type)
        # Identity rather than dropping the modules keeps one forward for "pre" and "none".
        make_norm = (lambda: RMSNorm(d_model=d_model, device=device)) if norm != "none" else nn.Identity
        self.ln1 = make_norm()
        self.ln2 = make_norm()

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        if self.norm == "post":  # the original Transformer: normalize after each residual add
            z = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(z + self.ffn(z))
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))
