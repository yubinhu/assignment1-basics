import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device=None, 
        dtype: torch.dtype=None,
    ):
        super().__init__()
        std = 1
        self.embedding = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean = 0,
                std = std,
                a = -3 * std,
                b = 3 * std
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]