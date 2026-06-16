import torch
from typing import Tuple


def split_coords(p: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    assert len(p.shape) == 1
    return (p[i] for i in range(len(p)))


def combine_coords(*coords: torch.Tensor) -> torch.Tensor:
    for coord in coords:
        assert len(coord.shape) == 0

    p = torch.zeros((len(coords),))
    for i, coord in enumerate(coords):
        p[i] = coord

    return p
