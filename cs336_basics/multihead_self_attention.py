import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)
        self.rope: RoPE = None
        if theta is not None:
            self.rope = RoPE(theta, self.d_head, max_seq_len, device=device)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        sequence_length = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split into heads
        q = rearrange(
            q,
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head",
            num_heads=self.num_heads,
            d_head=self.d_head,
        )
        k = rearrange(
            k,
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head",
            num_heads=self.num_heads,
            d_head=self.d_head,
        )
        v = rearrange(
            v,
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head",
            num_heads=self.num_heads,
            d_head=self.d_head,
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        self_attn_mask = ~torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        attn_out = scaled_dot_product_attention(q, k, v, self_attn_mask)

        attn_out = rearrange(
            attn_out,
            "... num_heads sequence_length d_head -> ... sequence_length (num_heads d_head)",
            num_heads=self.num_heads,
        )
        return self.output_proj(attn_out)
