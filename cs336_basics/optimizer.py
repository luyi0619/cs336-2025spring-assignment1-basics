from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                
                if t == 1:
                    m = (1 - beta1) * grad
                    v = (1 - beta2) * grad * grad
                else:
                    m = beta1 * state["m"] + (1 - beta1) * grad
                    v = beta2 * state["v"] + (1 - beta2) * grad * grad
                
                at = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= at * m  / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay *  p.data
                
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
  if it < warmup_iters:
    return it / warmup_iters * max_learning_rate
  
  if it <= cosine_cycle_iters:
    return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters)*math.pi)) * (max_learning_rate - min_learning_rate)
  
  return min_learning_rate
