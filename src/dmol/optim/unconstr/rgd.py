import torch

from dmol.diff_mfld.connection.methods.methods import ExpMapMethod
from dmol.diff_mfld.field.field_types import ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import MetricField, TangentConnection
from dmol.optim.methods import Retraction
from dmol.optim.unconstr.result import UnconstrResult


def rgd[M: Manifold](
    f: ScalarField[M],
    p0: Point[M] | torch.Tensor,
    metric: MetricField[M],
    conn: TangentConnection[M] | None = None,
    retr: Retraction[M] = ExpMapMethod.DEFAULT,
    tol: float = 1e-3,
    max_iters: int = 100,
    save_hist: bool = False,
    show_debug: bool = False,
    *,
    damp: float = 0.1,
) -> UnconstrResult[M]:
    if not ScalarField[f.bundle.base].compatible_field(f):
        raise ValueError("f must be a scalar field")

    if conn is None:
        conn = metric.levi_civita()
    f_diff = conn.total_covar(f)

    p = Point[f.bundle.base](p0)
    f_val = f(p)

    if show_debug:
        print(f"[rgd] initial: p={p.p}, f={f_val.components}")

    f_hist = [f_val.components.item()] if save_hist else None
    p_hist = [p.p] if save_hist else None

    success = False
    i: int = 0
    for i in range(max_iters):
        if show_debug:
            print(f"[rgd] i={i}, p={p.p}, f={f_val.components}")

        if success:
            break

        metric_tens = metric(p)
        f_grad = metric_tens.sharp(f_diff(p))

        p_next = retr(p, -damp * f_grad, conn)
        f_val_next = f(p_next)

        if torch.linalg.norm(p_next.p - p.p, ord=torch.inf) <= tol:
            if show_debug:
                print("[rgd] succeeded")
            success = True

        p = p_next
        f_val = f_val_next

        if save_hist:
            f_hist.append(f_val.components.item())  # type: ignore
            p_hist.append(p.p)  # type: ignore

    if show_debug:
        print(f"[rgd] final: p={p.p}, f={f_val.components}")

    result = UnconstrResult(success=success, num_iters=i, p=p, f=f_val)
    if save_hist:
        result.add_hist(f_hist, p_hist)  # type: ignore
    return result
