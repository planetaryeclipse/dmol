import torch

from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec

# approximation terms


def _f1_geod(v: torch.Tensor):
    return v


def _f2_geod(v: torch.Tensor, conns: torch.Tensor):
    return -torch.einsum("kij,i,j", conns, v, v)


def _f3_geod(v: torch.Tensor, dot_v: torch.Tensor, conns: torch.Tensor, conns_partials: torch.Tensor):
    return -(
        torch.einsum("kijl,l,i,j->k", conns_partials, v, v, v)
        + torch.einsum("kij,i,j->k", conns, dot_v, v)
        + torch.einsum("kij,i,j->k", conns, v, dot_v)
    )


def _f4_geod(
    v: torch.Tensor,
    dot_v: torch.Tensor,
    dot_dot_v: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
):

    return -(
        # first term
        torch.einsum("kijlr,r,l,i,j->k", conns_sec_partials, v, v, v, v)
        + torch.einsum("kijl,l,i,j->k", conns_partials, dot_v, v, v)
        + torch.einsum("kijl,l,i,j->k", conns_partials, v, dot_v, v)
        + torch.einsum("kijl,l,i,j->k", conns_partials, v, v, dot_v)
        # second term
        + torch.einsum("kijl,l,i,j->k", conns_partials, v, dot_v, v)
        + torch.einsum("kij,i,j->k", conns, dot_dot_v, v)
        + torch.einsum("kij,i,j->k", conns, dot_v, dot_v)
        # third term
        + torch.einsum("kijl,l,i,j->k", conns_partials, v, v, dot_v)
        + torch.einsum("kij,i,j->k", conns, dot_v, dot_v)
        + torch.einsum("kij,i,j->k", conns, v, dot_dot_v)
    )


def approx_exp_map[M: Manifold](p: Point[M] | torch.Tensor, v: Vec[M], conn: Connection[M], approx_order=1) -> Point[M]:
    p = Point[v.bundle.base](p)
    Vec[v.bundle.base].validate_tensor(v)

    if approx_order < 1:
        raise ValueError("approximate order must be at least 1")

    p_tens = p.p.detach()
    v_tens = v.components.detach()

    # implemented in each separate case for readability purposes
    if approx_order == 1:
        q = p_tens + v_tens
    elif approx_order == 2:
        conns = conn.coeffs(p)

        dot_v = _f2_geod(v_tens, conns)

        q = p_tens + v_tens + 1.0 / 2.0 * dot_v
    elif approx_order == 3:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)

        dot_v = _f2_geod(v_tens, conns)
        dot_dot_v = _f3_geod(v_tens, dot_v, conns, conns_partials)

        q = p_tens + v_tens + 1.0 / 2.0 * dot_v + 1.0 / 6.0 * dot_dot_v
    elif approx_order == 4:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)
        conns_sec_partials = conn.partials(p, 2)

        dot_v = _f2_geod(v_tens, conns)
        dot_dot_v = _f3_geod(v_tens, dot_v, conns, conns_partials)
        dot_dot_dot_v = _f4_geod(v_tens, dot_v, dot_dot_v, conns, conns_partials, conns_sec_partials)

        q = p_tens + v_tens + 1.0 / 2.0 * dot_v + 1 / 6.0 * dot_dot_v + 1 / 24.0 * dot_dot_dot_v
    else:
        raise NotImplementedError()
    return Point[v.bundle](q)


def _approx_log_o1(q: torch.Tensor, p: torch.Tensor):
    v_1 = q - p
    return v_1


def _approx_log_o2(q: torch.Tensor, p: torch.Tensor, conns: torch.Tensor):
    v_1 = q - p
    dot_v_1 = _f2_geod(v_1, conns)

    v_2 = q - p - 1.0 / 2.0 * dot_v_1
    return v_2


def _approx_log_o3(q: torch.Tensor, p: torch.Tensor, conns: torch.Tensor, conns_partials: torch.Tensor):
    v_1 = q - p
    dot_v_1 = _f2_geod(v_1, conns)

    v_2 = q - p - 1.0 / 2.0 * dot_v_1
    dot_v_2 = _f2_geod(v_2, conns)
    dot_dot_v_2 = _f3_geod(v_1, dot_v_1, conns, conns_partials)

    v_3 = q - p - 1.0 / 2.0 * dot_v_2 - 1.0 / 6.0 * dot_dot_v_2
    return v_3


def _approx_log_o4(
    q: torch.Tensor,
    p: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
):
    v_1 = q - p
    dot_v_1 = _f2_geod(v_1, conns)
    dot_dot_v_1 = _f3_geod(v_1, dot_v_1, conns, conns_partials)

    v_2 = q - p - 1.0 / 2.0 * dot_v_1
    dot_v_2 = _f2_geod(v_2, conns)
    dot_dot_v_2 = _f3_geod(v_1, dot_v_1, conns, conns_partials)

    v_3 = q - p - 1.0 / 2.0 * dot_v_2 - 1.0 / 6.0 * dot_dot_v_2
    dot_v_3 = _f2_geod(v_3, conns)
    dot_dot_v_3 = _f3_geod(v_2, dot_v_2, conns, conns_partials)
    dot_dot_dot_v_3 = _f4_geod(v_2, dot_v_1, dot_dot_v_1, conns, conns_partials, conns_sec_partials)

    v_4 = q - p - 1.0 / 2.0 * dot_v_3 - 1.0 / 6.0 * dot_dot_v_3 - 1.0 / 24.0 * dot_dot_dot_v_3
    return v_4


def approx_log_map[M: Manifold](
    p: Point[M] | torch.Tensor,
    q: Point[M] | torch.Tensor,
    conn: Connection[M],
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
        v = _approx_log_o2(q_tens, p_tens, conns)
    elif approx_order == 3:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p)

        v = _approx_log_o3(q_tens, p_tens, conns, conns_partials)
    elif approx_order == 4:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p)
        conns_sec_partials = conn.partials(p, 2)

        v = _approx_log_o4(q_tens, p_tens, conns, conns_partials, conns_sec_partials)
    else:
        raise NotImplementedError()
    return Vec[conn.bundle.base](v)
