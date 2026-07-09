from typing import Callable

import torch
import numpy as np

from dmol.diff_mfld.bundle.tensor import Scalar, Vec
from dmol.diff_mfld.connection.tangent import TangentConnection
from dmol.diff_mfld.field.field_types import ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import Metric, MetricField
from dmol.optim.unconstr.result import UnconstrResult

from scipy.optimize import minimize, NonlinearConstraint


def _tr_subproblem_cost(
    eta: np.ndarray, f: float, f_grad: np.ndarray, g: np.ndarray, h: Callable[[np.ndarray], np.ndarray] = lambda v: v
) -> float:
    value = f + f_grad @ g @ eta + 0.5 * h(eta) @ g @ eta
    return value.item()


def _tr_subproblem_constr(
    eta: np.ndarray,
    g: np.ndarray,
    radius: float,
) -> float:
    value = eta @ g @ eta - radius**2
    return value.item()


def _tr_subproblem[M: Manifold](
    f: Scalar[M],
    f_grad: Vec[M],
    g: Metric[M],
    radius: float,
    h: Callable[[Vec[M]], Vec[M]] = lambda p: p,  # symmetric
    tol: float = 1e-3,
    max_iters: int = 100,
) -> Vec[M]:
    f_val = f.components.item()
    f_grad_vec = f_grad.components.numpy()
    g_mat = g.components.numpy()
    h_wrapper = lambda v: h(Vec[f.bundle.base](torch.from_numpy(v))).components.numpy()

    eta_guess = np.zeros_like(f_grad_vec)
    result = minimize(
        fun=lambda eta: _tr_subproblem_cost(eta, f_val, f_grad_vec, g_mat, h_wrapper),
        constraints=[
            NonlinearConstraint(fun=lambda eta: _tr_subproblem_constr(eta, g_mat, radius), lb=-np.inf, ub=0.0)
        ],
        x0=eta_guess,
        tol=tol,
        options={"maxiter": max_iters},
    )

    eta = Vec[f.bundle.base](torch.from_numpy(result.x))
    return eta


def rtr[M: Manifold](
    f: ScalarField[M],
    p0: Point[M] | torch.Tensor,
    metric: MetricField[M],
    conn: TangentConnection[M] | None = None,
    retr: Callable[[Point[M], Vec[M]], Point[M]] | None = None,
    radius_max: float = 0.5,
    radius_start: float = 0.1,  # in (0, radius_max)
    quality_step_thresh: float = 0.15,  # in [0, 0.25]
    h: Callable[[Vec[M]], Vec[M]] = lambda p: p,  # symmetric
    tol: float = 1e-3,
    radius_eps: float = 1e-6,
    quality_eps: float = 1e-6,
    default_retr_damp: float = 0.9,
    max_iters: int = 1000,
    save_hist: bool = False,
) -> UnconstrResult:

    if conn is None:
        conn = metric.levi_civita()
    if retr is None:
        # if damping >= 1 then may be oscillatory and never converge
        retr = lambda p, v: conn.exp(p, default_retr_damp * v)[0]
    f_diff = conn.total_covar(f)
    h_wrapper = lambda v: h(Vec[f.bundle.base](torch.from_numpy(v))).components.numpy()

    p = Point[f.bundle.base](p0)
    f_val = f(p)
    radius = radius_start

    f_hist = [f_val.components.item()] if save_hist else None
    p_hist = [p.p] if save_hist else None

    success = False
    i: int = 0
    for i in range(max_iters):
        if success:
            break

        # solve the trust-region subproblem
        metric_tens = metric(p)
        f_diff_cov = f_diff(p)
        f_grad_vec = metric_tens.sharp(f_diff_cov)

        eta = _tr_subproblem(
            f_val,
            f_grad_vec,
            metric_tens,
            radius,
            h,
            tol,
            max_iters,
        )

        p_retr = retr(p, eta)  # cache as expensive

        # update confidence
        f_np = f_val.components.item()
        f_grad_np = f_grad_vec.components.numpy()
        g_np = metric_tens.components.numpy()
        eta_np = eta.components.numpy()

        quality = (f_np - f(p_retr).components.item()) / (
            _tr_subproblem_cost(np.zeros_like(eta_np), f_np, f_grad_np, g_np, h_wrapper)
            - _tr_subproblem_cost(eta_np, f_np, f_grad_np, g_np, h_wrapper)
            + quality_eps  # prevent division by zero
        )

        # perform any updates if criterion are met
        radius_next = radius
        if quality < 0.25:
            radius_next = 0.25 * radius
        elif quality > 0.75 and abs(torch.norm(eta.components) - radius) < radius_eps:
            radius_next = min(2 * radius, radius_max)

        p_next = p
        if quality > quality_step_thresh:
            p_next = p_retr
        f_val_next = f(p_next)

        print(
            f"p={p_next.p}, f={f_val_next.components}, f_grad={f_diff_cov.components}, radius={radius}, quality={quality}, eta={eta_np}, eta_norm={np.linalg.norm(eta_np)}"
        )

        if torch.linalg.norm(p_next.p - p.p, ord=torch.inf) <= tol:
            print(f"succeeded")
            success = True

        radius = radius_next
        p = p_next
        f_val = f_val_next

        if save_hist:
            f_hist.append(f_val.components.item())  # type: ignore
            p_hist.append(p.p)  # type: ignore

    result = UnconstrResult(success=success, p=p, f=f_val, num_iters=i)
    if save_hist:
        result.add_hist(f_hist, p_hist)  # type: ignore
    return result
