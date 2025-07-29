import sys
import os

from typing import IO, BinaryIO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    m = model.state_dict()
    o = optimizer.state_dict()
    torch.save((m, o, iteration), out)
    
    
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    (m, o, iteration) = torch.load(src)
    model.load_state_dict(m)
    optimizer.load_state_dict(o)
    return iteration