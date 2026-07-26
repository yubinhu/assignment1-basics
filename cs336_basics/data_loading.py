import numpy as np
import torch


def load_data(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, next-token target) sequences from a 1D array of token IDs.

    A start index i gives inputs dataset[i : i + context_length] and targets
    dataset[i + 1 : i + 1 + context_length], so i must stay below
    len(dataset) - context_length for the shifted window to be in bounds.
    """
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length)
    # (batch_size, 1) + (context_length,) broadcasts to (batch_size, context_length)
    indices = starts[:, None] + offsets

    # Fancy-indexing a memmap copies only the sampled windows into memory.
    x = torch.from_numpy(dataset[indices].astype(np.int64)).to(device)
    y = torch.from_numpy(dataset[indices + 1].astype(np.int64)).to(device)
    return x, y
