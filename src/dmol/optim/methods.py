from typing import Callable, Concatenate, Sequence

import torch

from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.connection.methods.methods import Distance
from dmol.diff_mfld.connection.tangent import TangentConnection
from dmol.diff_mfld.field.field_types import ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import MetricField
from dmol.optim.constr.result import ConstrResult
from dmol.optim.unconstr.result import UnconstrResult

type Retraction[M: Manifold] = Callable[[Point[M], Vec[M], TangentConnection[M]], Point[M]]

type UnconstrOptimFn[M: Manifold] = Callable[
    Concatenate[
        ScalarField[M],
        Point[M] | torch.Tensor,
        MetricField[M],
        TangentConnection[M] | None,
        Retraction[M],
        float,  # tolerance
        int,  # max iterations
        bool,  # save history
        bool,  # show debug
        ...,  # other method-specific arguments
    ],
    UnconstrResult[M],
]

type ConstrOptimFn[M: Manifold] = Callable[
    Concatenate[
        ScalarField[M],
        Sequence[ScalarField[M]],  # ineqs
        Sequence[ScalarField[M]],  # eqs
        Point[M] | torch.Tensor,
        MetricField[M],
        TangentConnection[M] | None,
        Retraction[M],
        Distance[M] | None,
        float,  # tolerance
        int,  # max iterations
        bool,  # save history
        bool,  # show debug
        ...,  # other method-specific arguments
    ],
    ConstrResult[M],
]
