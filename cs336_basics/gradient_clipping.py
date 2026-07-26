from typing import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    
    grads = [p.grad for p in parameters if p.grad is not None]
    l2_norm = 0
    for grad in grads:
        l2_norm += torch.sum(grad**2)
    l2_norm = torch.sqrt(l2_norm)
    if l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad * max_l2_norm / (l2_norm + 1e-6)