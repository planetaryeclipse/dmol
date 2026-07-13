from typing import Callable

import numpy as np
import torch

from scipy.integrate import solve_ivp

from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.curve import Curve
from dmol.diff_mfld.mfld import Manifold


def _parl_transp_vec_ivp_fn(
    t,
    y: np.ndarray,
    coeffs_fn: Callable[[np.ndarray], np.ndarray],
    curve_fn: Callable[[float], tuple[np.ndarray, np.ndarray]],
):
    p, v = curve_fn(t)
    u = y
    conn_coeffs = coeffs_fn(p)

    du = -np.einsum("i,j,kij->k", v, u, conn_coeffs)
    return du


def _get_numpy_pos_vel_from_curve(t, curve: Curve) -> tuple[np.ndarray, np.ndarray]:
    p, v = curve.sample(t)
    p_numpy = p.p.detach().numpy()
    v_numpy = v.components.detach().numpy()
    return p_numpy, v_numpy


def ivp_parl_transp_vec[M: Manifold](u: Vec[M], curve: Curve[M], conn: Connection[M], method="Radau") -> Vec[M]:
    # numpy is needed for use in scipy methods
    coeffs_np = lambda p: conn._eval(torch.from_numpy(p)).detach().numpy()
    curve_fn = lambda t: _get_numpy_pos_vel_from_curve(t, curve=curve)
    result = solve_ivp(
        _parl_transp_vec_ivp_fn,
        curve.interval,
        u.components.detach().numpy(),
        method=method,
        args=(coeffs_np, curve_fn),
    )

    w = result.y[:, -1]
    return Vec[u.bundle.base](torch.from_numpy(w))
