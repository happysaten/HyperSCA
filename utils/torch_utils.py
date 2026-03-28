"""Common PyTorch tensor utilities: normalization, Hamming weight/distance, and correlation computation."""

import torch
from typing import Literal
from torch import Tensor
from aes import HW_TABLE

HW_TABLE_TENSOR = torch.tensor(HW_TABLE, dtype=torch.uint8, device="cuda")


# Normalize a tensor and return a new tensor.
def standardization(tensor: Tensor, dim: int | None = None) -> Tensor:
    """Return a new tensor normalized along dim; dim=None normalizes the entire tensor."""
    keepdim = dim is not None
    mean = tensor.mean(dim=dim, keepdim=keepdim)
    std = tensor.std(dim=dim, keepdim=keepdim).clamp_min(1e-12)
    return tensor.sub(mean).div(std)


# Normalize a tensor in place.
def standardization_(tensor: Tensor, dim: int | None = None) -> Tensor:
    """Normalize the tensor in place and return it; dim=None means global normalization."""
    keepdim = dim is not None
    mean = tensor.mean(dim=dim, keepdim=keepdim)
    std = tensor.std(dim=dim, keepdim=keepdim).clamp_min_(1e-12)
    return tensor.sub_(mean).div_(std)


# Compute the Hamming weight of each tensor element.
def _calc_hw(v: Tensor) -> Tensor:
    """Return the Hamming weight of each element from the precomputed table (uint8 tensor)."""
    return HW_TABLE_TENSOR[v]


# Compute distances between two sets of data, supporting multiple distance types.
def _calc_dist(
    m1: Tensor,
    m2: Tensor,
    dist_type: Literal["hd", "hm", "l2", "l1", "hybrid"],
) -> Tensor:
    """Compute an element-wise distance matrix. dist_type: 'hd'/'hm'/'l1'/'l2'/'hybrid'."""
    match dist_type:
        case "hd":
            return _calc_hw(m1 ^ m2).float()
        case "hm":
            return torch.abs(_calc_hw(m1) - _calc_hw(m2)).float()
        case "l1":
            return torch.abs(m1.float() - m2.float())
        case "l2":
            return (m1.float() - m2.float()).square_()
        case "hybrid":
            dist_hd, dist_hw = (
                _calc_dist(m1, m2, dist_type="hd"),
                _calc_dist(m1, m2, dist_type="hm"),
            )
            return standardization_(dist_hd) + standardization_(dist_hw)
        case _:
            raise ValueError(f"Unknown dist_type: {dist_type}")


# Generate a distance-space matrix and normalize it.
def generate_dist_space(
    dist_type: Literal["hd", "hm", "l2", "l1", "hybrid"], dist_dim: int = 256
) -> Tensor:
    """Generate and return a row-normalized distance-space matrix of size dist_dim x dist_dim."""
    a = torch.arange(dist_dim, dtype=torch.uint8, device="cuda")
    b = a.view(-1, 1)
    dist_space = _calc_dist(a, b, dist_type)
    dist_space = dist_space.log1p_().log1p_()
    return standardization_(dist_space, dim=1)


# Compute the correlation between the input and the distance space (compiled optimization).
@torch.compile
def calc_dist_corr(input: Tensor, dist: Tensor) -> Tensor:
    """Return the dot product of the normalized input and dist (correlation matrix)."""
    return torch.mm(standardization_(input, dim=1), dist.type_as(input))


# Compute the correlation coefficient between two tensors (legacy implementation).
def calc_corr_old(m1: Tensor, m2: Tensor, dim: int) -> Tensor:
    """Legacy correlation implementation: compute a Pearson-like correlation along the specified dim and return the result."""
    if m1.ndim != m2.ndim:
        raise ValueError("m1 and m2 must have the same number of dimensions")
    if m1.shape[dim] != m2.shape[dim]:
        raise ValueError(
            "m1 and m2 must have the same size along the specified dimension"
        )

    mean1 = m1.mean(dim=dim, keepdim=True)
    mean2 = m2.mean(dim=dim, keepdim=True)
    std1 = m1.std(dim=dim) + 1e-12
    std2 = m2.std(dim=dim) + 1e-12

    corr = torch.mean((m1 - mean1) * (m2 - mean2), dim=dim) / (std1 * std2)
    return corr
