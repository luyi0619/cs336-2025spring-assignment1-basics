import math
import torch
from collections.abc import Iterable
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " vocab_size"]) -> Float[Tensor, ""]:
    numerator = torch.gather(inputs, -1, targets.unsqueeze(-1)) # log and exp cancels out
    denominator = torch.logsumexp(inputs, dim=-1)
    return torch.mean(denominator - numerator)

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