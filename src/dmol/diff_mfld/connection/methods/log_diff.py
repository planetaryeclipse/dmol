import torch
import numpy as np

from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Tensor, Vec
from dmol.diff_mfld.bundle.vector_bundle import TensorBundle

from dmol.diff_mfld.connection.methods.geod_approx import (
    _f1_geod,
    _f2_geod,
    _f3_geod,
    _f4_geod,
    _approx_log_o1,
    _approx_log_o2,
    _approx_log_o3,
    _approx_log_o4,
    approx_log_map,
)

from dmol.diff_mfld.connection.methods.geod_ivp_bvp import bvp_log_map
from dmol.diff_mfld.riemann import EuclideanMetricField, MetricLambdaField


def _f1_log_diff(q: torch.Tensor, p: torch.Tensor):
    n = len(q)
    return -torch.eye(n)


def _f2_log_diff(
    v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    v_partials: torch.Tensor,  # cached quantities
):
    return -(
        torch.einsum("kijp,i,j->kp", conns_partials, v, v)
        + torch.einsum("kij,ip,j->kp", conns, v_partials, v)
        + torch.einsum("kij,i,jp->kp", conns, v, v_partials)
    )


def _f3_log_diff(
    v: torch.Tensor,
    dot_v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
    v_partials: torch.Tensor,  # cached quantities
    dot_v_partials: torch.Tensor,
):
    return -(
        # first term
        torch.einsum("kijlp,l,i,j->kp", conns_sec_partials, v, v, v)
        + torch.einsum("kijl,lp,i,j->kp", conns_partials, v_partials, v, v)
        + torch.einsum("kijl,l,ip,j->kp", conns_partials, v, v_partials, v)
        + torch.einsum("kijl,l,i,jp->kp", conns_partials, v, v, v_partials)
        # second term
        + torch.einsum("kijp,i,j->kp", conns_partials, dot_v, v)
        + torch.einsum("kij,ip,j->kp", conns, dot_v_partials, v)
        + torch.einsum("kij,i,jp->kp", conns, dot_v, v_partials)
        # third term
        + torch.einsum("kijp,i,j->kp", conns_partials, v, dot_v)
        + torch.einsum("kij,ip,j->kp", conns, v_partials, dot_v)
        + torch.einsum("kij,i,jp->kp", conns, v, dot_v_partials)
    )


def _f4_log_diff(
    v: torch.Tensor,
    dot_v: torch.Tensor,
    dot_dot_v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
    conns_thd_partials: torch.Tensor,
    v_partials: torch.Tensor,  # cached quantities
    dot_v_partials: torch.Tensor,
    dot_dot_v_partials: torch.Tensor,
):
    return -(
        # first term
        torch.einsum("kijlrp,r,l,i,j->kp", conns_thd_partials, v, v, v, v)
        + torch.einsum("kijlr,rp,l,i,j->kp", conns_sec_partials, v_partials, v, v, v)
        + torch.einsum("kijlr,r,lp,i,j->kp", conns_sec_partials, v, v_partials, v, v)
        + torch.einsum("kijlr,r,l,ip,j->kp", conns_sec_partials, v, v, v_partials, v)
        + torch.einsum("kijlr,r,l,i,jp->kp", conns_sec_partials, v, v, v, v_partials)
        # second term
        + torch.einsum("kijlp,l,i,j->kp", conns_sec_partials, dot_v, v, v)
        + torch.einsum("kijl,lp,i,j->kp", conns_partials, dot_v_partials, v, v)
        + torch.einsum("kijl,l,ip,j->kp", conns_partials, dot_v, v_partials, v)
        + torch.einsum("kijl,l,i,jp->kp", conns_partials, dot_v, v, v_partials)
        # third term
        + torch.einsum("kijlp,l,i,j->kp", conns_sec_partials, v, dot_v, v)
        + torch.einsum("kijl,lp,i,j->kp", conns_partials, v_partials, dot_v, v)
        + torch.einsum("kijl,l,ip,j->kp", conns_partials, v, dot_v_partials, v)
        + torch.einsum("kijl,l,i,jp->kp", conns_partials, v, dot_v, v_partials)
        # fourth term
        + torch.einsum("kijlp,l,i,j->kp", conns_sec_partials, v, v, dot_v)
        + torch.einsum("kijl,lp,i,j->kp", conns_partials, v_partials, v, dot_v)
        + torch.einsum("kijl,l,ip,j->kp", conns_partials, v, v_partials, dot_v)
        + torch.einsum("kijl,l,i,jp->kp", conns_partials, v, v, dot_v_partials)
        # fifth term
        + torch.einsum("kijlp,l,i,j->kp", conns_sec_partials, v, dot_v, v)
        + torch.einsum("kijl,lp,i,j->kp", conns_partials, v_partials, dot_v, v)
        + torch.einsum("kijl,l,ip,j->kp", conns_partials, v, dot_v_partials, v)
        + torch.einsum("kijl,l,i,jp->kp", conns_partials, v, dot_v, v_partials)
        # sixth term
        + torch.einsum("kijp,i,j->kp", conns_partials, dot_dot_v, v)
        + torch.einsum("kij,ip,j->kp", conns, dot_dot_v_partials, v)
        + torch.einsum("kij,i,jp->kp", conns, dot_dot_v, v_partials)
        # seventh term
        + torch.einsum("kijp,i,j->kp", conns_partials, dot_v, dot_v)
        + torch.einsum("kij,ip,j->kp", conns, dot_v_partials, dot_v)
        + torch.einsum("kij,i,jp->kp", conns, dot_v, dot_v_partials)
        # eighth term
        + torch.einsum("kijlp,l,i,j->kp", conns_sec_partials, v, v, dot_v)
        + torch.einsum("kijl,lp,i,j->kp", conns_partials, v_partials, v, dot_v)
        + torch.einsum("kijl,l,ip,j->kp", conns_partials, v, v_partials, dot_v)
        + torch.einsum("kijl,l,i,jp->kp", conns_partials, v, v, dot_v_partials)
        # ninth term
        + torch.einsum("kijp,i,j->kp", conns_partials, dot_v, dot_v)
        + torch.einsum("kij,ip,j->kp", conns, dot_v_partials, dot_v)
        + torch.einsum("kij,i,jp->kp", conns, dot_v, dot_v_partials)
        # tenth term
        + torch.einsum("kijp,i,j->kp", conns_partials, v, dot_dot_v)
        + torch.einsum("kij,ip,j->kp", conns, v_partials, dot_dot_v)
        + torch.einsum("kij,i,jp->kp", conns, v, dot_dot_v_partials)
    )


def _approx_log_covar_o1(q: torch.Tensor, p: torch.Tensor):
    n = len(q)
    return -torch.eye(n)


def _approx_log_covar_o2(
    q: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
):
    n = len(q)

    v_partials_1 = -torch.eye(n)
    dot_v_partials_1 = _f2_log_diff(v, conns, conns_partials, v_partials_1)

    v_partials_2 = -torch.eye(n) - 1.0 / 2.0 * dot_v_partials_1
    return v_partials_2


def _approx_log_covar_o3(
    q: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
    dot_v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
):
    n = len(q)

    v_partials_1 = -torch.eye(n)
    dot_v_partials_1 = _f2_log_diff(v, conns, conns_partials, v_partials_1)

    v_partials_2 = -torch.eye(n) - 1.0 / 2.0 * dot_v_partials_1
    dot_v_partials_2 = _f2_log_diff(v, conns, conns_partials, v_partials_2)
    dot_dot_v_partials_2 = _f3_log_diff(
        v,
        dot_v,
        conns,
        conns_partials,
        conns_sec_partials,
        v_partials_1,
        dot_v_partials_1,
    )

    v_partials_3 = -torch.eye(n) - 1.0 / 2.0 * dot_v_partials_2 - 1.0 / 6.0 * dot_dot_v_partials_2
    return v_partials_3


def _approx_log_covar_o4(
    q: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
    dot_v: torch.Tensor,
    dot_dot_v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
    conns_thd_partials: torch.Tensor,
):
    n = len(q)

    v_partials_1 = -torch.eye(n)
    dot_v_partials_1 = _f2_log_diff(v, conns, conns_partials, v_partials_1)
    dot_dot_v_partials_1 = _f3_log_diff(
        v, dot_v, conns, conns_partials, conns_sec_partials, v_partials_1, dot_v_partials_1
    )

    v_partials_2 = -torch.eye(n) - 1.0 / 2.0 * dot_v_partials_1
    dot_v_partials_2 = _f2_log_diff(v, conns, conns_partials, v_partials_2)
    dot_dot_v_partials_2 = _f3_log_diff(
        v,
        dot_v,
        conns,
        conns_partials,
        conns_sec_partials,
        v_partials_1,
        dot_v_partials_1,
    )

    v_partials_3 = -torch.eye(n) - 1.0 / 2.0 * dot_v_partials_2 - 1.0 / 6.0 * dot_dot_v_partials_2
    dot_v_partials_3 = _f2_log_diff(v, conns, conns_partials, v_partials_3)
    dot_dot_v_partials_3 = _f3_log_diff(
        v,
        dot_v,
        conns,
        conns_partials,
        conns_sec_partials,
        v_partials_2,
        dot_v_partials_2,
    )
    dot_dot_dot_v_partials_3 = _f4_log_diff(
        v,
        dot_v,
        dot_dot_v,
        conns,
        conns_partials,
        conns_sec_partials,
        conns_thd_partials,
        v_partials_2,
        dot_v_partials_1,
        dot_dot_v_partials_1,
    )

    v_partials_4 = (
        -torch.eye(n)
        - 1.0 / 2.0 * dot_v_partials_3
        - 1.0 / 6.0 * dot_dot_v_partials_3
        - 1.0 / 24.0 * dot_dot_dot_v_partials_3
    )
    return v_partials_4


def approx_log_covar[M: Manifold](
    p: Point[M] | torch.Tensor, q: Point[M] | torch.Tensor, v: Vec[M], conn: Connection[M], approx_order=1
) -> Tensor[TensorBundle[1, 1]]:
    p = Point[conn.bundle.base](p)
    q = Point[conn.bundle.base](q)

    if approx_order < 1:
        raise ValueError("approximate order must be at least 1")

    p_tens, q_tens = p.p.detach(), q.p.detach()
    v_tens = v.components

    n = len(p_tens)
    conns = conn.coeffs(p)

    # implemented in each separate case for readability purposes
    if approx_order == 1:
        v_partials = _approx_log_covar_o1(q_tens, p_tens)
    elif approx_order == 2:
        conns_partials = conn.partials(p)
        v_partials = _approx_log_covar_o2(q_tens, p_tens, v_tens, conns, conns_partials)
    elif approx_order == 3:
        conns_partials = conn.partials(p)
        conns_sec_partials = conn.partials(p, 2)

        dot_v = _f2_geod(v_tens, conns)

        v_partials = _approx_log_covar_o3(
            q_tens,
            p_tens,
            v_tens,
            dot_v,
            conns,
            conns_partials,
            conns_sec_partials,
        )
    elif approx_order == 4:
        conns_partials = conn.partials(p)
        conns_sec_partials = conn.partials(p, 2)
        conns_thd_partials = conn.partials(p, 3)

        dot_v = _f2_geod(v_tens, conns)
        dot_dot_v = _f3_geod(v_tens, dot_v, conns, conns_partials)

        v_partials = _approx_log_covar_o4(
            q_tens,
            p_tens,
            v_tens,
            dot_v,
            dot_dot_v,
            conns,
            conns_partials,
            conns_sec_partials,
            conns_thd_partials,
        )
    else:
        raise NotImplementedError()

    v_covar = v_partials + torch.einsum("k,ijk->ij", v.components, conns)
    return Tensor[TensorBundle[1, 1]][conn.bundle.base](v_covar)
