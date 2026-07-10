from typing import Sequence

import torch

from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.connection.covar_diff import FieldCustomCovar
from dmol.diff_mfld.field.base import Field
from dmol.diff_mfld.field.field_types import ScalarField


def coord_repr(
    repr: torch.Tensor | float | Sequence[torch.Tensor | float] | Sequence[Sequence[torch.Tensor | float]],
) -> torch.Tensor:
    # scalar-valued
    if type(repr) is torch.Tensor:
        return repr
    elif type(repr) is float:
        return torch.tensor(repr)
    # vector-valued
    elif type(repr[0]) is torch.Tensor or type(repr[0]) is float:  # type: ignore
        vec_repr: Collection[torch.Tensor | float, ...] = repr  # type: ignore
        n = len(vec_repr)

        vec = torch.zeros((n,))
        for i in range(n):
            vec[i] = vec_repr[i]
        return vec
    # matrix-valued
    else:
        mat_repr: Collection[Collection[torch.Tensor | float, ...], ...] = repr  # type: ignore
        n = len(mat_repr)
        m = len(mat_repr[0])

        mat = torch.zeros((n, m))
        for i in range(n):
            for j in range(m):
                mat[i, j] = mat_repr[i][j]
        return mat
