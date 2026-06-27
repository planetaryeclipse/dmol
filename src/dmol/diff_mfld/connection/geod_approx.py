import torch

from typing import Callable

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec

import dmol.diff_mfld.connection.connection as conn

# approximation terms


def _f1_geod(v: torch.Tensor):
    return v


def _f2_geod(v: torch.Tensor, conns: torch.Tensor):
    return -torch.einsum("kij,i,j", conns, v, v)


def _f3_geod(v: torch.Tensor, conns: torch.Tensor, conns_partials: torch.Tensor):
    dot_v = _f2_geod(v, conns)
    return -(
        torch.einsum("kijl,l,i,j->k", conns_partials, v, v, v)
        + torch.einsum("kij,i,j->k", conns, dot_v, v)
        + torch.einsum("kij,i,j->k", conns, v, dot_v)
    )


def _f4_geod(v: torch.Tensor, conns: torch.Tensor, conns_partials: torch.Tensor, conns_sec_partials: torch.Tensor):
    dot_v = _f2_geod(v, conns)
    return (
        # first term
        torch.einsum("kijlr,r,l,i,j->k", conns_sec_partials, v, v, v, v)
        + torch.einsum("kijl,l,i,j->k", conns_partials, dot_v, v, v)
        + torch.einsum("kijl,l,i,j->k", conns_partials, v, dot_v, v)
        + torch.einsum("kijl,l,i,j->k", conns_partials, v, v, dot_v)
        # second term
        + torch.einsum("kijl,l,ist,s,t,j->k", conns_partials, v, conns, v, v, v)
        + torch.einsum("kij,istl,l,s,t,j->k", conns, conns_partials, v, v, v, v)
        + torch.einsum("kij,ist,s,t,j->k", conns, conns, dot_v, v, v)
        + torch.einsum("kij,ist,s,t,j->k", conns, conns, v, dot_v, v)
        + torch.einsum("kij,ist,s,t,j->k", conns, conns, v, v, dot_v)
        # third term
        + torch.einsum("kijl,l,i,jst,s,t->k", conns_partials, v, v, conns, v, v)
        + torch.einsum("kij,i,jst,s,t->k", conns, dot_v, conns, v, v)
        + torch.einsum("kij,i,jstl,l,s,t->k", conns, v, conns_partials, v, v, v)
        + torch.einsum("kij,i,jst,s,t->k", conns, v, conns, dot_v, v)
        + torch.einsum("kij,i,jst,s,t->k", conns, v, conns, v, dot_v)
    )


def approx_exp_map[M: Manifold](
    p: Point[M] | torch.Tensor, v: Vec[M], conn: conn.TangentConnection[M], approx_order=1
) -> Point[M]:
    p = Point[v.bundle.base](p)
    Vec[v.bundle.base].validate_tensor(v)

    if approx_order < 1:
        raise ValueError("approximate order must be at least 1")

    p_tens = p.p.detach()
    v_tens = v.components.detach()

    # implemented in each separate case for readability purposes
    if approx_order == 1:
        q = p_tens + _f1_geod(v_tens)
        pass
    elif approx_order == 2:
        conns = conn.coeffs(p)
        q = p_tens + _f1_geod(v_tens) + 1.0 / 2.0 * _f2_geod(v_tens, conns)
    elif approx_order == 3:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)
        q = (
            p_tens
            + _f1_geod(v_tens)
            + 1.0 / 2.0 * _f2_geod(v_tens, conns)
            + 1.0 / 6.0 * _f3_geod(v_tens, conns, conns_partials)
        )
    elif approx_order == 4:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)
        conns_sec_partials = conn.partials(p, 2)
        q = (
            p_tens
            + _f1_geod(v_tens)
            + 0.5 * _f2_geod(v_tens, conns)
            + 1 / 6.0 * _f3_geod(v_tens, conns, conns_partials)
            + 1 / 24.0 * _f4_geod(v_tens, conns, conns_partials, conns_sec_partials)
        )
    else:
        raise NotImplementedError()
    return Point[v.bundle](q)


def _approx_log_o1(q: torch.Tensor, p: torch.Tensor):
    return q - p


def _approx_log_o2(q: torch.Tensor, p: torch.Tensor, v: torch.Tensor, conns: torch.Tensor):
    return q - p - 1.0 / 2.0 * _f2_geod(v, conns)


def _approx_log_o3(
    q: torch.Tensor, p: torch.Tensor, v: torch.Tensor, conns: torch.Tensor, conns_partials: torch.Tensor
):
    return q - p - 1.0 / 2.0 * _f2_geod(v, conns) - 1.0 / 6.0 * _f3_geod(v, conns, conns_partials)


def _approx_log_o4(
    q: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_seq_partials: torch.Tensor,
):
    return (
        q
        - p
        - 1.0 / 2.0 * _f2_geod(v, conns)
        - 1.0 / 6.0 * _f3_geod(v, conns, conns_partials)
        - 1.0 / 24.0 * _f4_geod(v, conns, conns_partials, conns_seq_partials)
    )


def approx_log_map[M: Manifold](
    p: Point[M] | torch.Tensor,
    q: Point[M] | torch.Tensor,
    conn: conn.TangentConnection[M],
    approx_order=1,
) -> Vec[M]:
    p = Point[conn.bundle.base](p)
    q = Point[conn.bundle.base](q)

    if approx_order < 1:
        raise ValueError("approximate order must be at least 1")

    p_tens, q_tens = p.p.detach(), q.p.detach()

    # implemented in each separate case for readability purposes
    if approx_order == 1:
        v = _approx_log_o1(q_tens, p_tens)
    elif approx_order == 2:
        conns = conn.coeffs(p)
        v = _approx_log_o1(q_tens, p_tens)
        v = _approx_log_o2(q_tens, p_tens, v, conns)
    elif approx_order == 3:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p)
        v = _approx_log_o1(q_tens, p_tens)
        v = _approx_log_o2(q_tens, p_tens, v, conns)
        v = _approx_log_o3(q_tens, p_tens, v, conns, conns_partials)
    elif approx_order == 4:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p)
        conns_seq_partials = conn.partials(p, 2)
        v = _approx_log_o1(q_tens, p_tens)
        v = _approx_log_o2(q_tens, p_tens, v, conns)
        v = _approx_log_o3(q_tens, p_tens, v, conns, conns_partials)
        v = _approx_log_o4(q_tens, p_tens, v, conns, conns_partials, conns_seq_partials)
    else:
        raise NotImplementedError()
    return Vec[conn.bundle.base](v)
