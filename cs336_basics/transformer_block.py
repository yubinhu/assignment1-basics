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
    ):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device)
        self.ln1 = RMSNorm(d_model=d_model, device=device)
        self.ln2 = RMSNorm(d_model=d_model, device=device)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))
