import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from einops import rearrange

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None, device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_head = d_model // num_heads
        self.Q = Linear(d_model, d_model, device=device)
        self.K = Linear(d_model, d_model, device=device)
        self.V = Linear(d_model, d_model, device=device)
        self.O = Linear(d_model, d_model, device=device)
        self.rope : RoPE = None
        if theta is not None:
            self.rope = RoPE(theta, self.d_head, max_seq_len, device=device)


    def forward(self, x: Float[Tensor, " ... sequence_length d_model"], token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_model"]:

        sequence_length = x.shape[-2]

        Q = self.Q(x) 
        K = self.K(x)
        V = self.V(x)

        # split into heads
        Q = rearrange(
            Q, 
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head", num_heads=self.num_heads, d_head=self.d_head
        )
        K = rearrange(
            K, 
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head", num_heads=self.num_heads, d_head=self.d_head
        )
        V = rearrange(
            V, 
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head", num_heads=self.num_heads, d_head=self.d_head
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        self_attn_mask = ~torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool),
            diagonal=1
        )

        attn_out = scaled_dot_product_attention(Q, K, V, self_attn_mask)

        attn_out = rearrange(
            attn_out, 
            "... num_heads sequence_length d_head -> ... sequence_length (num_heads d_head)", num_heads=self.num_heads
        )
        out = self.O(attn_out)
        return out
        

