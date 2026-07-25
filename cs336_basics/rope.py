import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
from torch import Tensor

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        assert d_k % 2 == 0
        
        i_mat = rearrange(
            torch.arange(max_seq_len, device=device),
            "max_seq_len -> max_seq_len 1"
        )
        k_mat = rearrange(
            torch.arange(1, d_k // 2+1, device=device),
            "half -> 1 half"
        )
        
        theta_mat = i_mat / (theta ** ((2 * k_mat - 2) / d_k)) # broadcast -> (max_seq_len, half)
        cos_mat = torch.cos(theta_mat)
        sin_mat = torch.sin(theta_mat) # (max_seq_len, half)

        self.register_buffer("cos_mat", cos_mat, persistent=False)
        self.register_buffer("sin_mat", sin_mat, persistent=False)

    def forward(
        self, 
        x: Float[Tensor, " ... seq_len d_k"], 
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:

        rearranged = rearrange(x, "... seq_len (half two) -> ... seq_len half two", two=2)
        left, right = rearranged[..., 0], rearranged[..., 1]  # ... seq_len half
        cos_rel = self.cos_mat[token_positions] # ... seq_len, half
        sin_rel = self.sin_mat[token_positions] # ... seq_len, half
        top = cos_rel * left - sin_rel * right
        bottom = sin_rel * left + cos_rel * right # ... seq_len, half
        out = rearrange(torch.stack([top, bottom], dim=-1), "... seq_len half two -> ... seq_len (half two)")
        return out


if __name__ == "__main__":
    d_k = 8
    test_tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32)
    # seq_len, d_k
    print(test_tensor)
    rearranged = rearrange(test_tensor, "... seq_len (half two) -> ... seq_len half two", two=2)
    print(rearranged, rearranged.shape)
    # rearranged = rearrange(test_tensor, "... (two half) -> ... half two", two=2)
    # print(rearranged, rearranged.shape)

    # rearranged = rearrange(test_tensor, "... (half two) -> ... two half", two=2)
    # print(rearranged, rearranged.shape)

    max_seq_len = 10

    i_mat = rearrange(
        torch.arange(max_seq_len),
        "max_seq_len -> max_seq_len 1"
    )
    k_mat = rearrange(
        torch.arange(1, d_k // 2+1),
        "half -> 1 half"
    )
    
    th = 1000
    theta_mat = i_mat / (th ** ((2 * k_mat - 2) / d_k)) # broadcast -> (max_seq_len, half)
    cos_mat = torch.cos(theta_mat)
    sin_mat = torch.sin(theta_mat) # (max_seq_len, half)

    token_positions = torch.tensor([1]) # ... seq_len

    left, right = rearranged[..., 0], rearranged[..., 1]  # ... seq_len half
    print("left, right", left.shape, right.shape)
    cos_rel = cos_mat[token_positions] # ... seq_len, half
    sin_rel = sin_mat[token_positions] # ... seq_len, half
    top = cos_rel * left - sin_rel * right
    bottom = sin_rel * left + cos_rel * right # ... seq_len, half
    res = rearrange(torch.stack([top, bottom], dim=-1), "... seq_len half two -> ... seq_len (half two)")
    print(res, res.shape)