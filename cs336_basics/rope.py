import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
from torch import Tensor

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        assert d_k % 2 == 0

        

    def forward(
        self, 
        x: Float[Tensor, " ... seq_len d_k"], 
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        pass


if __name__ == "__main__":
    d_k = 8
    test_tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32)
    # seq_len, d_k
    print(test_tensor)
    rearranged = rearrange(test_tensor, "... (half two) -> ... half two", two=2)
    print(rearranged, rearranged.shape)

    # rearranged = rearrange(test_tensor, "... (two half) -> ... half two", two=2)
    # print(rearranged, rearranged.shape)

    # rearranged = rearrange(test_tensor, "... (half two) -> ... two half", two=2)
    # print(rearranged, rearranged.shape)

    cos = 1
    sin = 1

    def mat_ik(i, k):
        return torch.tensor(
            [
                [cos*i/k, -sin*i/k],
                [sin*i/k, cos*i/k]
            ],
            dtype=torch.float32
        )

    def mat_i(i):
        return torch.stack(
            [mat_ik(i, k) for k in range(1, d_k // 2+1)]
        )
    print("mat_1", mat_i(1), mat_i(1).shape)

    max_seq_len = 10
    mat_full = torch.stack(
        [mat_i(i) for i in range(max_seq_len)]
    )
    print(mat_full.shape) # max_seq_len, d_k // 2, 2, 2

    token_positions = torch.tensor([1]) # ... seq_len

    relevant_mat = mat_full[token_positions] # seq_len, dk // 2, 2, 2

    res = einsum(
        rearranged,
        relevant_mat,
        "... half two, ... half other_two two -> ... half other_two"
    )
    print(res, res.shape)

    res = rearrange(res, "... half two -> ... (half two)")
    print(res, res.shape)