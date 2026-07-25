from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    c = in_features.max(dim=dim, keepdim=True).values
    normalized = in_features - c
    exp = torch.exp(normalized)
    return exp / exp.sum(dim=dim, keepdim=True)


if __name__ == "__main__":
    in_features = torch.randn(4, 5)
    print(in_features, in_features.shape)
    c = in_features.max(dim=1, keepdim=True).values
    print(c, c.shape)
    normalized = in_features - c
    print(normalized, normalized.shape)