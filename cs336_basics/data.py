
import numpy.typing as npt
import numpy as np
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_start_index = dataset.shape[0] - context_length
    start_indices = np.random.randint(0, max_start_index, size=batch_size)
    indices = start_indices[:, None] + np.arange(context_length)
    
    batch = torch.tensor(dataset[indices], device = device)
    targets = torch.tensor(dataset[indices + 1], device = device)
    return (batch, targets)