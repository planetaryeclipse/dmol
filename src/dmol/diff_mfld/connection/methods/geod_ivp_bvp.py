import numpy as np
import torch

from typing import Callable
from scipy.integrate import solve_ivp, solve_bvp

from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.curve import Curve


def _exp_map_ivp_fn(t, y: np.ndarray, n: int, coeffs_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    p, v = y[:n], y[n:]
    conn_coeffs = coeffs_fn(p)

    dot_p = v
    dot_v = -np.einsum("kij,i,j->k", conn_coeffs, v, v)

    return np.concat((dot_p, dot_v))


def ivp_exp_map[M: Manifold](
    p: Point[M] | torch.Tensor, v: Vec[M], conn: Connection[M], method="Radau"
) -> tuple[Point[M], Curve[M]]:
    p = Point[v.bundle.base](p)
    Vec[v.bundle.base].validate_tensor(v)

    n = p.manifold.dim

    # numpy is needed for use in scipy methods
    coeffs_np = lambda p: conn._eval(torch.from_numpy(p)).detach().numpy()
    result = solve_ivp(
        _exp_map_ivp_fn,
        [0.0, 1.0],
        np.concat((p.p.detach().numpy(), v.components.detach().numpy())),
        method=method,
        dense_output=True,
        args=(n, coeffs_np),
    )

    q = result.y[:n, -1]
    t_hist = result.t
    p_hist = result.y[:n, :]
    v_hist = result.y[n:, :]

    point = Point[p.manifold](torch.from_numpy(q))
    curve = Curve[p.manifold](t_hist, p_hist, v_hist)
    return point, curve


def _exp_map_ivp_fn_batched(
    _t, y: np.ndarray, n: int, coeffs_batched_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    p_batched, v_batched = y[:n, :], y[n:, :]  # (n, samples), (n, samples)
    conns_batched = coeffs_batched_fn(p_batched)  # (n, n, n, samples)

    dot_p_batched = v_batched
    dot_v_batched = -np.einsum("kijl,il,jl->kl", conns_batched, v_batched, v_batched)

    return np.concat((dot_p_batched, dot_v_batched))


def _exp_map_ivp_bc_fn(
    ya: np.ndarray,
    yb: np.ndarray,
    n: int,
    p: np.ndarray,
    q: np.ndarray,
):
    pos_a, pos_b = ya[:n], yb[:n]
    return np.hstack((pos_a - p, pos_b - q))


def _coeffs_np_batched(p_batched: np.ndarray, coeffs_np: Callable[[np.ndarray], np.ndarray]):
    p_indiv = [p_batched[:, i] for i in range(p_batched.shape[1])]
    conns_indiv = [coeffs_np(p)[:, :, :, np.newaxis] for p in p_indiv]
    return np.concat(conns_indiv, axis=3)


def bvp_log_map[M: Manifold](
    p: Point[M] | torch.Tensor, q: Point[M] | torch.Tensor, conn: Connection[M]
) -> tuple[Vec[M], Curve[M]]:
    p = Point[conn.bundle.base](p)
    q = Point[conn.bundle.base](q)

    p_numpy, q_numpy = p.p.detach().numpy(), q.p.detach().numpy()

    t_initial_mesh = np.linspace(0.0, 1.0)

    p_initial = np.linspace(p_numpy, q_numpy).T
    v_initial = np.tile(q_numpy - p_numpy, (len(t_initial_mesh), 1)).T
    y_initial = np.vstack((p_initial, v_initial))

    coeffs_np = lambda p: conn._eval(torch.from_numpy(p)).detach().numpy()
    coeffs_np_batched = lambda p_batched: _coeffs_np_batched(p_batched, coeffs_np)

    n = p.manifold.dim

    # TODO: review tolerance and max nodes for use in solving
    result = solve_bvp(
        lambda t, y: _exp_map_ivp_fn_batched(t, y, n, coeffs_np_batched),
        lambda ya, yb: _exp_map_ivp_bc_fn(ya, yb, n, p_numpy, q_numpy),
        t_initial_mesh,
        y_initial,
        # tol=1e-7,
        # max_nodes=3000,
    )

    t_hist = result.x
    p_hist = result.y[:n, :]
    v_hist = result.y[n:, :]

    if not result.success:
        raise ValueError("failed to find solution to bvp log map")

    vec = Vec[conn.bundle.base](torch.from_numpy(result.y[n:, 0]))
    curve = Curve[conn.bundle.base](t_hist, p_hist, v_hist)
    return vec, curve
