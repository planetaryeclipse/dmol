import torch

from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec

from dmol.diff_mfld.connection.methods.geod_approx import _f2_geod, _f3_geod, _f4_geod

# approximate methods for parallel transport along a geodesic


def _f1_parl_transp_vec(v: torch.Tensor, u: torch.Tensor, conns: torch.Tensor) -> torch.Tensor:
    return -torch.einsum("i,j,kij->k", v, u, conns)


def _f2_parl_transp_vec(
    v: torch.Tensor,
    dot_v: torch.Tensor,
    u: torch.Tensor,
    dot_u: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
) -> torch.Tensor:
    return -(
        torch.einsum("i,j,kij->k", dot_v, u, conns)
        + torch.einsum("i,j,kij->k", v, dot_u, conns)
        + torch.einsum("i,j,kijl,l->k", v, u, conns_partials, v)
    )


def _f3_parl_transp_vec(
    v: torch.Tensor,
    dot_v: torch.Tensor,
    dot_dot_v: torch.Tensor,
    u: torch.Tensor,
    dot_u: torch.Tensor,
    dot_dot_u: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
) -> torch.Tensor:
    return -(
        # first term
        torch.einsum("i,j,kij->k", dot_dot_v, u, conns)
        + torch.einsum("i,j,kij->k", dot_v, dot_u, conns)
        + torch.einsum("i,j,kijl,l->k", dot_v, u, conns_partials, v)
        # second term
        + torch.einsum("i,j,kij->k", dot_v, dot_u, conns)
        + torch.einsum("i,j,kij->k", v, dot_dot_u, conns)
        + torch.einsum("i,j,kijl,l->k", v, dot_u, conns_partials, v)
        # third term
        + torch.einsum("i,j,kijl,l->k", dot_v, u, conns_partials, v)
        + torch.einsum("i,j,kijl,l->k", v, dot_u, conns_partials, v)
        + torch.einsum("i,j,kijlr,r,l->k", v, u, conns_sec_partials, v, v)
        + torch.einsum("i,j,kijl,l->k", v, u, conns_partials, dot_v)
    )


def _f4_parl_transp_vec(
    v: torch.Tensor,
    dot_v: torch.Tensor,
    dot_dot_v: torch.Tensor,
    dot_dot_dot_v: torch.Tensor,
    u: torch.Tensor,
    dot_u: torch.Tensor,
    dot_dot_u: torch.Tensor,
    dot_dot_dot_u: torch.Tensor,
    conns: torch.Tensor,
    conns_partials: torch.Tensor,
    conns_sec_partials: torch.Tensor,
    conns_thd_partials: torch.Tensor,
) -> torch.Tensor:
    return -(
        # first term
        torch.einsum("i,j,kij->k", dot_dot_dot_v, u, conns)
        + torch.einsum("i,j,kij->k", dot_dot_v, dot_u, conns)
        + torch.einsum("i,j,kijl,l->k", dot_dot_v, u, conns_partials, v)
        # second term
        + torch.einsum("i,j,kij->k", dot_dot_v, dot_u, conns)
        + torch.einsum("i,j,kij->k", dot_v, dot_dot_u, conns)
        + torch.einsum("i,j,kijl,l->k", dot_v, dot_u, conns_partials, v)
        # third term
        + torch.einsum("i,j,kijl,l->k", dot_dot_v, u, conns_partials, v)
        + torch.einsum("i,j,kijl,l->k", dot_v, dot_u, conns_partials, v)
        + torch.einsum("i,j,kijlr,r,l->k", dot_v, u, conns_sec_partials, v, v)
        + torch.einsum("i,j,kijl,l->k", dot_v, u, conns_partials, dot_v)
        # fourth term
        + torch.einsum("i,j,kij->k", dot_dot_v, dot_u, conns)
        + torch.einsum("i,j,kij->k", dot_v, dot_dot_u, conns)
        + torch.einsum("i,j,kijl,l->k", dot_v, dot_u, conns_partials, v)
        # fifth term
        + torch.einsum("i,j,kij->k", dot_v, dot_dot_u, conns)
        + torch.einsum("i,j,kij->k", v, dot_dot_dot_u, conns)
        + torch.einsum("i,j,kijl,l->k", v, dot_dot_u, conns_partials, v)
        # sixth term
        + torch.einsum("i,j,kijl,l->k", dot_v, dot_u, conns_partials, v)
        + torch.einsum("i,j,kijl,l->k", v, dot_dot_u, conns_partials, v)
        + torch.einsum("i,j,kijlr,r,l->k", v, dot_u, conns_sec_partials, v, v)
        + torch.einsum("i,j,kijl,l->k", v, dot_u, conns_partials, dot_v)
        # seventh term
        + torch.einsum("i,j,kijl,l->k", dot_dot_v, u, conns_partials, v)
        + torch.einsum("i,j,kijl,l->k", dot_v, dot_u, conns_partials, v)
        + torch.einsum("i,j,kijlr,r,l->k", dot_v, u, conns_sec_partials, v, v)
        + torch.einsum("i,j,kijl,l->k", dot_v, u, conns_partials, dot_v)
        # eighth term
        + torch.einsum("i,j,kijl,l->k", dot_v, dot_u, conns_partials, v)
        + torch.einsum("i,j,kijl,l->k", v, dot_dot_u, conns_partials, v)
        + torch.einsum("i,j,kijlr,r,l->k", v, v, conns_sec_partials, v, v)
        + torch.einsum("i,j,kijl,l->k", v, dot_u, conns_partials, dot_v)
        # ninth term
        + torch.einsum("i,j,kijlr,r,l->k", dot_v, u, conns_sec_partials, v, v)
        + torch.einsum("i,j,kijlr,r,l->k", v, dot_u, conns_sec_partials, v, v)
        + torch.einsum("i,j,kijlrp,p,r,l->k", v, u, conns_thd_partials, v, v, v)
        + torch.einsum("i,j,kijlr,r,l->k", v, u, conns_sec_partials, dot_v, v)
        + torch.einsum("i,j,kijlr,r,l->k", v, u, conns_sec_partials, v, dot_v)
        # tenth term
        + torch.einsum("i,j,kijl,l->k", dot_v, u, conns_partials, dot_v)
        + torch.einsum("i,j,kijl,l->k", v, dot_u, conns_partials, dot_v)
        + torch.einsum("i,j,kijlr,r,l->k", v, u, conns_sec_partials, v, dot_v)
        + torch.einsum("i,j,kijl,l->k", v, u, conns_partials, dot_dot_v)
    )


def approx_parl_transp_vec[M: Manifold](
    u: Vec[M], p: Point[M] | torch.Tensor, v: Vec[M], conn: Connection[M], approx_order=1
) -> Vec[M]:
    Vec[conn.bundle.base].validate_tensor(u)
    Vec[conn.bundle.base].validate_tensor(v)
    p = Point[conn.bundle.base](p)

    if approx_order < 1:
        raise ValueError("approximate order must be at least 1")

    v_tens = v.components.detach()
    u_tens = u.components.detach()

    # implemented in each separate case for readability purposes
    if approx_order == 1:
        conns = conn.coeffs(p)

        dot_u = _f1_parl_transp_vec(v_tens, u_tens, conns)

        w = u_tens + dot_u
    elif approx_order == 2:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)

        dot_v = _f2_geod(v_tens, conns)

        dot_u = _f1_parl_transp_vec(v_tens, u_tens, conns)
        dot_dot_u = _f2_parl_transp_vec(v_tens, dot_v, u_tens, dot_u, conns, conns_partials)

        w = u_tens + dot_u + 1.0 / 2.0 * dot_dot_u
    elif approx_order == 3:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)
        conns_sec_partials = conn.partials(p, 2)

        dot_v = _f2_geod(v_tens, conns)
        dot_dot_v = _f3_geod(v_tens, dot_v, conns, conns_partials)

        dot_u = _f1_parl_transp_vec(v_tens, u_tens, conns)
        dot_dot_u = _f2_parl_transp_vec(v_tens, dot_v, u_tens, dot_u, conns, conns_partials)
        dot_dot_dot_u = _f3_parl_transp_vec(
            v_tens, dot_v, dot_dot_v, u_tens, dot_u, dot_dot_u, conns, conns_partials, conns_sec_partials
        )

        w = u_tens + dot_u + 1.0 / 2.0 * dot_dot_u + 1.0 / 6.0 * dot_dot_dot_u
    elif approx_order == 4:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)
        conns_sec_partials = conn.partials(p, 2)
        conns_thd_partials = conn.partials(p, 3)

        dot_v = _f2_geod(v_tens, conns)
        dot_dot_v = _f3_geod(v_tens, dot_v, conns, conns_partials)
        dot_dot_dot_v = _f4_geod(v_tens, dot_v, dot_dot_v, conns, conns_partials, conns_sec_partials)

        dot_u = _f1_parl_transp_vec(v_tens, u_tens, conns)
        dot_dot_u = _f2_parl_transp_vec(v_tens, dot_v, u_tens, dot_u, conns, conns_partials)
        dot_dot_dot_u = _f3_parl_transp_vec(
            v_tens, dot_v, dot_dot_v, u_tens, dot_u, dot_dot_u, conns, conns_partials, conns_sec_partials
        )
        dot_dot_dot_dot_u = _f4_parl_transp_vec(
            v_tens,
            dot_v,
            dot_dot_v,
            dot_dot_dot_v,
            u_tens,
            dot_u,
            dot_dot_u,
            dot_dot_dot_u,
            conns,
            conns_partials,
            conns_sec_partials,
            conns_thd_partials,
        )

        w = u_tens + dot_u + 1.0 / 2.0 * dot_dot_u + 1.0 / 6.0 * dot_dot_dot_u + 1.0 / 24.0 * dot_dot_dot_dot_u
    else:
        raise NotImplementedError()
    return Vec[conn.bundle.base](w)
