import torch
from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if not grads: # No gradients to clip
      return
     
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g, ord=2) for g in grads]))
     
    if total_norm > max_l2_norm:
      clip_coeff = max_l2_norm / (total_norm + 1e-6) # Add small epsilon for stability
      for p in parameters:
        if p.grad is not None:
          p.grad = p.grad * clip_coeff