from typing import Callable

import torch

from torch.func import jacrev

from dmol.diff_mfld.util import split_coords


def check_field_expr_callable_for_gradient(
    fn: Callable[[*tuple[torch.Tensor, ...]], torch.Tensor],
    coord_dim: int,
    single_arg=False,
    num_samples=20,
    coord_mean=2.0,
    coord_std=5.0,
):
    p_dist = torch.distributions.MultivariateNormal(
        coord_mean * torch.ones((coord_dim,)), coord_std**2 * torch.eye(coord_dim)
    )

    fn_single_arg = fn if single_arg else lambda p: fn(*split_coords(p))
    fn_grad = jacrev(fn_single_arg)

    # samples the initial point
    p_initial = p_dist.sample()
    initial_val = fn_single_arg(p_initial)

    for _ in range(num_samples):
        p_sample = p_dist.sample()
        val = fn_single_arg(p_sample)
        grad = fn_grad(p_sample)

        # if the value has changed we expect the gradient to also be nonzero
        if not torch.allclose(initial_val, val):
            if torch.allclose(grad, torch.zeros_like(grad)):
                raise ValueError(
                    "zero gradient detected thereby suggesting probable gradient loss in field expression callable"
                )

        # evaluates between a number of randomly sampled points
        initial_val = val
