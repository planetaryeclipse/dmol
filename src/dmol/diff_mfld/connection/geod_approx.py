import torch

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.connection.connection import TangentConnection

# approximation terms


def _f0(y: torch.Tensor):
    return y


def _f1(y: torch.Tensor, conns: torch.Tensor):
    return -torch.einsum("kij,i,j", conns, y, y)


def _f2(y: torch.Tensor, conns: torch.Tensor, conns_partials: torch.Tensor):
    return (
        -torch.einsum("kijl,l,i,j->k", conns_partials, y, y, y)
        + torch.einsum("kij,ist,s,t,j->k", conns, conns, y, y, y)
        + torch.einsum("kij,i,jst,s,t->k", conns, y, conns, y, y)
    )


def _f3(y: torch.Tensor, conns: torch.Tensor, conns_partials: torch.Tensor, conns_sec_partials: torch.Tensor):
    return (
        # deriv. of first term
        -torch.einsum("kijlr,r,l,i,j->k", conns_sec_partials, y, y, y, y)
        + torch.einsum("kijl,lst,s,t,i,j->k", conns_partials, conns, y, y, y, y)
        + torch.einsum("kijl,l,ist,s,t,j->k", conns_partials, y, conns, y, y, y)
        + torch.einsum("kijl,l,i,ist,s,t->k", conns_partials, y, y, conns, y, y)
        # deriv. of second term
        + torch.einsum("kijl,l,ist,s,t,j->k", conns_partials, y, conns, y, y, y)
        + torch.einsum("kij,istl,l,s,t,j->k", conns, conns_partials, y, y, y, y)
        - torch.einsum("kij,ist,suv,u,v,t,j->k", conns, conns, conns, y, y, y, y)
        - torch.einsum("kij,ist,s,tuv,u,v,j->k", conns, conns, y, conns, y, y, y)
        - torch.einsum("kij,ist,s,t,juv,u,v->k", conns, conns, y, y, conns, y, y)
        # deriv. of third term
        - torch.einsum("kijl,l,i,ist,s,t->k", conns_partials, y, y, conns, y, y)
        - torch.einsum("kij,iuv,u,v,jst,s,t->k", conns, conns, y, y, conns, y, y)
        + torch.einsum("kij,i,jstl,l,s,t->k", conns, y, conns_partials, y, y, y)
        - torch.einsum("kij,i,jst,suv,u,v,t->k", conns, y, conns, conns, y, y, y)
        - torch.einsum("kij,i,tuv,u,v->k", conns, y, conns, y, y)
    )


def approx_exp_map[M: Manifold](
    p: Point[M] | torch.Tensor, v: Vec[M], conn: TangentConnection[M], approx_order=1
) -> Point[M]:
    p = Point[v.bundle.base](p)
    Vec[v.bundle.base].validate_tensor(v)

    if approx_order < 1:
        raise ValueError("approximate order must be at least 1")

    p_tens = p.p.detach()
    v_tens = v.components.detach()

    q = p_tens + _f0(v_tens)

    # implemented in each separate case for readability purposes
    if approx_order == 1:
        q = p_tens + _f0(v_tens)
        pass
    elif approx_order == 2:
        conns = conn.coeffs(p)
        q = p_tens + _f0(v_tens) + 0.5 * _f1(v_tens, conns)
    elif approx_order == 3:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)
        q = p_tens + _f0(v_tens) + 0.5 * _f1(v_tens, conns) + 1 / 6.0 * _f2(v_tens, conns, conns_partials)
    elif approx_order == 4:
        conns = conn.coeffs(p)
        conns_partials = conn.partials(p, 1)
        conns_sec_partials = conn.partials(p, 2)
        q = (
            p_tens
            + _f0(v_tens)
            + 0.5 * _f1(v_tens, conns)
            + 1 / 6.0 * _f2(v_tens, conns, conns_partials)
            + 1 / 24.0 * _f3(v_tens, conns, conns_partials, conns_sec_partials)
        )

    return Point[v.bundle](q)
