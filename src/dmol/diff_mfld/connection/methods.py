import torch

from enum import Enum, auto
from typing import Any

from dmol.diff_mfld.mfld import Point, Manifold
from dmol.diff_mfld.bundle.tensor import Tensor, Vec
from dmol.diff_mfld.bundle.vector_bundle import TensorBundle
from dmol.diff_mfld.connection.connection import TangentConnection
from dmol.diff_mfld.connection.geod_ivp_bvp import bvp_log_map
from dmol.diff_mfld.connection.geod_approx import approx_log_map
from dmol.diff_mfld.connection.log_diff import approx_log_covar


class _IncludesCurve(Enum):
    WITH_CURVE = auto()
    NO_CURVE = auto()


class LogMapMethod(Enum):
    DFEAULT = (bvp_log_map, {}, _IncludesCurve.WITH_CURVE)
    BVP = (bvp_log_map, {}, _IncludesCurve.WITH_CURVE)  # wrap in tuple to force execution through __call__
    APPROX_O1 = (approx_log_map, {"approx_order": 1}, _IncludesCurve.NO_CURVE)
    APPROX_O2 = (approx_log_map, {"approx_order": 2}, _IncludesCurve.NO_CURVE)
    APPROX_O3 = (approx_log_map, {"approx_order": 3}, _IncludesCurve.NO_CURVE)
    APPROX_O4 = (approx_log_map, {"approx_order": 4}, _IncludesCurve.NO_CURVE)

    def __call__(self, p: Point | torch.Tensor, q: Point | torch.Tensor, conn: TangentConnection) -> Vec:
        p = Point[conn.bundle.base](p)
        q = Point[conn.bundle.base](q)

        method, kwargs, includes_curve = self.value
        match includes_curve:
            case _IncludesCurve.WITH_CURVE:
                vec, _ = method(p, q, conn, **kwargs)  # type: ignore
                return vec
            case _IncludesCurve.NO_CURVE:
                vec = method(p, q, conn, **kwargs)
                return vec  # type: ignore
            case _:
                raise NotImplementedError()  # unreachable


class LogMapCovarMethod(Enum):
    DEFAULT = (approx_log_covar, {"approx_order": 4})
    APPROX_O1 = (approx_log_covar, {"approx_order": 1})
    APPROX_O2 = (approx_log_covar, {"approx_order": 2})
    APPROX_O3 = (approx_log_covar, {"approx_order": 3})
    APPROX_O4 = (approx_log_covar, {"approx_order": 4})

    def __call__(
        self, p: Point | torch.Tensor, q: Point | torch.Tensor, v: Vec, conn: TangentConnection
    ) -> Tensor[TensorBundle[1,]]:
        p = Point[conn.bundle.base](p)
        q = Point[conn.bundle.base](q)

        method, kwargs = self.value
        log_covar = method(p, q, v, conn, **kwargs)
        return log_covar
