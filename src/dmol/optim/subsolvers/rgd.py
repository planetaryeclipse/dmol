import torch

from dataclasses import dataclass
from typing import Callable

from dmol.diff_mfld.field.field_types import ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.field import Field
from dmol.diff_mfld.bundle.tensor import Vec, Scalar
from dmol.diff_mfld.riemann import Metric, MetricField, TangentConnection


@dataclass
class UnconstrResult[M: Manifold]:
    success: bool
    num_iters: int
    p: Point[M]
    f: Scalar[M]
    f_hist: torch.Tensor | None = None
    p_hist: torch.Tensor | None = None


def rgd[M: Manifold](
    f: Field[M],
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

    f_hist = [f] if save_hist else None
    p_hist = [p] if save_hist else None

    success = False
    i: int = 0
    for i in range(max_iters):
        if success:
            break

        metric_tensor: Metric[M] = metric(p)  # type: ignore
        f_grad = metric_tensor.sharp(f_diff(p))

        p_next, _ = conn.exp(p, -damp * f_grad)
        f_val_next = f(p)

        print(f"p={p_next.p}, f={f_val_next.components}, f_grad={f_grad.components}")

        if torch.linalg.norm(p_next.p - p.p, ord=torch.inf) <= tol:
            success = True

        p = p_next
        f_val = f_val_next

        # record the current step cost and position if enabled
        if save_hist:
            f_hist.append(f_val)  # type: ignore
            p_hist.append(p.p)  # type: ignore

    result = UnconstrResult(success=success, p=p, f=f_val, num_iters=i)
    if save_hist:
        num_samples = len(f_hist)  # type: ignore
        f_hist_tens = torch.zeros((num_samples,))
        p_hist_tens = torch.zeros((f.bundle.base.dim, num_samples))
        for i in range(num_samples):
            f_hist_tens[i] = f_hist[i]  # type: ignore
            p_hist_tens[i, :] = p_hist[i]  # type: ignore

        result.f_hist = f_hist_tens
        result.p_hist = p_hist_tens
    return result
