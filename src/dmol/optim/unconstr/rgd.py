import torch

from typing import Callable

from dmol.diff_mfld.field.field_types import ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.riemann import Metric, MetricField, TangentConnection
from dmol.optim.unconstr.result import UnconstrResult


def rgd[M: Manifold](
    f: ScalarField[M],
    p0: Point[M] | torch.Tensor,
    metric: MetricField[M],
    conn: TangentConnection[M] | None = None,
    retr: Callable[[Point[M], Vec[M]], Point[M]] | None = None,
    damp: float = 0.1,
    tol: float = 1e-3,
    max_iters: int = 1000,
    save_hist: bool = False,
) -> UnconstrResult:
    print(f.bundle)

    if not ScalarField[f.bundle.base].compatible_field(f):
        raise ValueError("f must be a scalar field")

    if conn is None:
        conn = metric.levi_civita()
    if retr is None:
        # sets the retraction as the exponential map (ignore curve ouput)
        retr = lambda p, v: conn.exp(p, v)[0]
    f_diff = conn.total_covar(f)

    p = Point[f.bundle.base](p0)
    f_val = f(p)

    print(f"initial: p={p.p}, f={f_val.components}, f_diff={f_diff(p).components}")

    f_hist = [f_val.components.item()] if save_hist else None
    p_hist = [p.p] if save_hist else None

    success = False
    i: int = 0
    for i in range(max_iters):
        if success:
            break

        metric_tens = metric(p)
        f_grad = metric_tens.sharp(f_diff(p))

        p_next, _ = conn.exp(p, -damp * f_grad)
        f_val_next = f(p_next)

        print(f"p={p_next.p}, f={f_val_next.components}, f_grad={f_grad.components}")

        if torch.linalg.norm(p_next.p - p.p, ord=torch.inf) <= tol:
            success = True

        p = p_next
        f_val = f_val_next

        # record the current step cost and position if enabled
        if save_hist:
            f_hist.append(f_val.components.item())  # type: ignore
            p_hist.append(p.p)  # type: ignore

    result = UnconstrResult(success=success, p=p, f=f_val, num_iters=i)
    if save_hist:
        result.add_hist(f_hist, p_hist)  # type: ignore
    return result
