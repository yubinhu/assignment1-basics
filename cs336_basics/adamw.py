from typing import Callable, Optional
from torch import nn, Tensor
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                lr_t = lr * (1 - beta2**t)**0.5 / (1 - beta1**t) # Not a real learning rate, just adding a correction factor
                m = beta1 * m + (1 - beta1) * p.grad
                v = beta2 * v + (1 - beta2) * p.grad**2
                p.data = p.data - lr * weight_decay * p.data - lr_t * m / (v**0.5 + eps)
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss