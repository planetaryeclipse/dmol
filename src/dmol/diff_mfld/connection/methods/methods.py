import torch

from enum import Enum, auto

from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.mfld import Point
from dmol.diff_mfld.bundle.tensor import Tensor, Vec
from dmol.diff_mfld.bundle.vector_bundle import TensorBundle
from dmol.diff_mfld.connection.methods.geod_ivp_bvp import bvp_log_map
from dmol.diff_mfld.connection.methods.geod_approx import approx_log_map
from dmol.diff_mfld.connection.methods.geod_log_diff import approx_log_covar


class _IncludesCurve(Enum):
    WITH_CURVE = auto()
    NO_CURVE = auto()


class LogMapMethod(Enum):
    DEFAULT = (bvp_log_map, {}, _IncludesCurve.WITH_CURVE)
    BVP = (bvp_log_map, {}, _IncludesCurve.WITH_CURVE)  # wrap in tuple to force execution through __call__
    APPROX_O1 = (approx_log_map, {"approx_order": 1}, _IncludesCurve.NO_CURVE)
    APPROX_O2 = (approx_log_map, {"approx_order": 2}, _IncludesCurve.NO_CURVE)
    APPROX_O3 = (approx_log_map, {"approx_order": 3}, _IncludesCurve.NO_CURVE)
    APPROX_O4 = (approx_log_map, {"approx_order": 4}, _IncludesCurve.NO_CURVE)

    def __call__(self, p: Point | torch.Tensor, q: Point | torch.Tensor, conn: Connection) -> Vec:
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
        self, p: Point | torch.Tensor, q: Point | torch.Tensor, v: Vec, conn: Connection
    ) -> Tensor[TensorBundle[1,]]:
        p = Point[conn.bundle.base](p)
        q = Point[conn.bundle.base](q)

        method, kwargs = self.value
        log_covar = method(p, q, v, conn, **kwargs)
        return log_covar
