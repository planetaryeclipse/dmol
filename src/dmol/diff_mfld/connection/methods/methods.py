from typing import Callable

import torch

from enum import Enum, auto

from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.connection.methods.geod_parl_transp import approx_parl_transp_vec
from dmol.diff_mfld.connection.methods.parl_transp import ivp_parl_transp_vec
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.connection.methods.geod_ivp_bvp import bvp_log_map, ivp_exp_map
from dmol.diff_mfld.connection.methods.geod_approx import approx_exp_map, approx_log_map
from dmol.diff_mfld.riemann import MetricField

# varous types of methods

type Exp[M: Manifold] = Callable[
    [
        Point[M] | torch.Tensor,
        Vec[M],
        Connection[M],
    ],
    Point[M],
]
type Log[M: Manifold] = Callable[
    [
        Point[M] | torch.Tensor,
        Point[M] | torch.Tensor,
        Connection[M],
    ],
    Vec[M],
]
type Distance[M: Manifold] = Callable[
    [
        Point[M] | torch.Tensor,
        Point[M] | torch.Tensor,
        MetricField[M],
        Connection[M] | None,
    ],
    float,
]
type GeodParlTransp[M: Manifold] = Callable[
    [
        Vec[M],
        Point[M] | torch.Tensor,
        Vec[M],
        Connection[M],
    ],
    Vec[M],
]

# wrapper types providing all implementations


class _IncludesCurve(Enum):
    WITH_CURVE = auto()
    NO_CURVE = auto()


class ExpMapMethod(Enum):
    DEFAULT = (ivp_exp_map, {}, _IncludesCurve.WITH_CURVE)
    IVP = (ivp_exp_map, {}, _IncludesCurve.WITH_CURVE)
    APPROX_O1 = (approx_exp_map, {"approx_order": 1}, _IncludesCurve.NO_CURVE)
    APPROX_O2 = (approx_exp_map, {"approx_order": 2}, _IncludesCurve.NO_CURVE)
    APPROX_O3 = (approx_exp_map, {"approx_order": 3}, _IncludesCurve.NO_CURVE)
    APPROX_O4 = (approx_exp_map, {"approx_order": 4}, _IncludesCurve.NO_CURVE)

    def __call__(self, p: Point | torch.Tensor, v: Vec, conn: Connection) -> Point:
        p = Point[conn.bundle.base](p)
        Vec[conn.bundle.base].validate_tensor(v)

        method, kwargs, includes_curve = self.value
        match includes_curve:
            case _IncludesCurve.WITH_CURVE:
                q, _ = method(p, v, conn, **kwargs)  # type: ignore
                return q
            case _IncludesCurve.NO_CURVE:
                q = method(p, v, conn, **kwargs)  # type: ignore
                return q  # type: ignore
            case _:
                raise NotImplementedError()  # unreachable


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


class DistanceMethod(Enum):
    DEFAULT = (LogMapMethod.DEFAULT,)
    BVP = (LogMapMethod.BVP,)
    APPROX_O1 = (LogMapMethod.APPROX_O1,)
    APPROX_O2 = (LogMapMethod.APPROX_O2,)
    APPROX_O3 = (LogMapMethod.APPROX_O3,)
    APPROX_O4 = (LogMapMethod.APPROX_O4,)

    def __call__(
        self,
        p: Point | torch.Tensor,
        q: Point | torch.Tensor,
        metric: MetricField,
        conn: Connection | None = None,
    ) -> float:
        (log_method,) = self.value
        if conn is None:
            conn = metric.levi_civita()
        v = log_method(p, q, conn)
        return metric(p).norm(v)


class GeodParlTranspMethod(Enum):
    DEFAULT = ()
    APPROX_O1 = (approx_parl_transp_vec, {"approx_order": 1})
    APPROX_O2 = (approx_parl_transp_vec, {"approx_order": 2})
    APPROX_O3 = (approx_parl_transp_vec, {"approx_order": 3})
    APPROX_O4 = (approx_parl_transp_vec, {"approx_order": 4})

    def __call__(self, u: Vec, p: Point | torch.Tensor, v: Vec, conn: Connection) -> Vec:
        match self:
            case self.DEFAULT:
                _, curve = ivp_exp_map(p, v, conn)
                w = ivp_parl_transp_vec(u, curve, conn)
                return w
            case _:
                method, kwargs = self.value
                w = method(u, p, v, conn, **kwargs)
                return w
