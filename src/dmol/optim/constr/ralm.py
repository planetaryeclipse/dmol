from typing import Callable, Concatenate, ParamSpec, Sequence
from math import sqrt

import torch

from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.connection.tangent import TangentConnection
from dmol.diff_mfld.field.field_types import FloatField, ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import MetricField
from dmol.optim.constr.result import ConstrResult
from dmol.optim.unconstr.rgd import UnconstrOptimFn, Retraction, rgd

type Distance[M: Manifold] = Callable[[Point[M], Point[M]], float]

P = ParamSpec("P")
type ConstrOptimFn[M: Manifold, **P] = Callable[
    Concatenate[
        ScalarField[M],
        Sequence[ScalarField[M]],  # ineqs
        Sequence[ScalarField[M]],  # eqs
        Point[M] | torch.Tensor,
        MetricField[M],
        TangentConnection[M] | None,
        Retraction[M] | None,
        Distance[M] | None,
        UnconstrOptimFn,  # subsolver method
        dict,  # subsolver args
        P,
    ]
]


def _default_dist[M: Manifold](
    p: Point[M] | torch.Tensor,
    q: Point[M] | torch.Tensor,
    conn: TangentConnection[M],
    metric: MetricField[M],
) -> float:
    v = conn.log(p, q)[0]
    return sqrt(metric(p).inner(v, v))


def _eval_constrs_tens[M: Manifold](p: Point[M] | torch.Tensor, constrs: Sequence[ScalarField[M]]) -> torch.Tensor:
    constr_vals = torch.zeros((len(constrs)))
    for i in range(len(constrs)):
        constr_vals[i] = constrs[i](p).components.item()
    return constr_vals


def ralm[M: Manifold](
    f: ScalarField[M],
    ineqs: Sequence[ScalarField[M]],
    eqs: Sequence[ScalarField[M]],
    p0: Point[M] | torch.Tensor,
    metric: MetricField[M],
    conn: TangentConnection[M] | None = None,
    retr: Retraction[M] | None = None,
    dist: Distance[M] | None = None,
    subsolver_method: UnconstrOptimFn = rgd,
    subsolver_args: dict = {},
    tol: float = 1e-3,
    max_iters: int = 1000,
    save_hist: bool = False,
    show_debug: bool = False,
    *,
    penalty_start: float = 0.1,
    penalty_growth: float = 1.1,  # > 1
    ineq_mult_start: float = 0.0,
    ineq_mults_min: float | torch.Tensor = -torch.inf,
    ineq_mults_max: float | torch.Tensor = torch.inf,
    eq_mult_start: float = 0.0,
    eq_mults_min: float | torch.Tensor = -torch.inf,
    eq_mults_max: float | torch.Tensor = torch.inf,
    subsolver_tol_start: float = 1e-1,
    subsolver_tol_min: float = 1e-3,
    subsolver_tol_decay: float = 0.5,
    subsolver_max_iters=100,
    ratio: float = 0.8,
) -> ConstrResult[M]:
    if not ScalarField[f.bundle.base].compatible_field(f):
        raise ValueError("f must be a scalar field")
    for ineq in ineqs:
        if not ScalarField[f.bundle.base].compatible_field(ineq):
            raise ValueError("all inequality constraints must be scalar fields")
    for eq in eqs:
        if not ScalarField[f.bundle.base].compatible_field(eq):
            raise ValueError("all equality constraints must be scalar fields")

    if conn is None:
        conn = metric.levi_civita()
    if retr is None:
        retr = lambda p, v: conn.exp(p, v)[0]
    if dist is None:
        dist = lambda p, q: _default_dist(p, q, conn, metric)

    num_ineqs = len(ineqs)
    num_eqs = len(eqs)
    is_constrained = num_ineqs > 0 or num_eqs > 0

    p = Point[f.bundle.base](p0)
    penalty = penalty_start
    ineq_mults = ineq_mult_start * torch.ones((num_ineqs,))
    eq_mults = eq_mult_start * torch.ones((num_eqs,))

    # setup parameterized fields (to be modified during optimization)
    penalty_field = FloatField[f.bundle.base](penalty)
    ineq_mult_fields = [FloatField[f.bundle.base](ineq_mults[i].item()) for i in range(num_ineqs)]
    eq_mult_fields = [FloatField[f.bundle.base](eq_mults[i].item()) for i in range(num_eqs)]

    # setup the augmented lagrangian
    # TODO: refactor composition to allow using ints to allow using sum directly (w/out the explicit float zero here)
    ineq_constrs = sum(
        ScalarField.max(0.0, ineq + mult / penalty_field) ** 2 for ineq, mult in zip(ineqs, ineq_mult_fields)
    )
    eq_constrs = sum((eq + mult / penalty_field) ** 2 for eq, mult in zip(eqs, eq_mult_fields))
    aug_lagr = f + penalty_field / 2.0 * (ineq_constrs + eq_constrs)

    subsolver_tol = subsolver_tol_start

    f_val = f(p)
    ineqs_eval = _eval_constrs_tens(p, ineqs)
    eqs_eval = _eval_constrs_tens(p, eqs)
    sigma: torch.Tensor

    if show_debug:
        print(f"[ralm] initial: p={p.p}, f={f_val.components}")

    f_hist = [f_val.components.item()] if save_hist else None
    ineqs_eval_hist = [ineqs_eval] if save_hist else None
    eqs_eval_hist = [eqs_eval] if save_hist else None
    p_hist = [p.p] if save_hist else None

    success = False
    i: int = 0
    for i in range(max_iters):
        if show_debug:
            print(f"[ralm] i={i}, p={p.p}, f={f_val.components}")

        if success:
            break

        result = subsolver_method(
            aug_lagr,
            p,
            metric,
            conn,
            retr,
            subsolver_tol,
            subsolver_max_iters,
            False,
            show_debug,
            **subsolver_args,
        )
        if not result.success:
            break

        p_next: Point = result.p  # type: ignore
        if dist(p, p_next) < tol:
            if show_debug:
                print("[ralm] succeeded")
            success = True

        next_subsolver_tol = max(subsolver_tol_min, subsolver_tol_decay * subsolver_tol)  # type: ignore
        if is_constrained:
            next_ineqs_eval = _eval_constrs_tens(p_next, ineqs)
            next_eqs_eval = _eval_constrs_tens(p_next, eqs)
            next_ineq_mults = torch.clip(
                ineq_mults + penalty * next_ineqs_eval,
                ineq_mults_min,  # type: ignore
                ineq_mults_max,  # type: ignore
            )
            next_eq_mults = torch.clip(
                eq_mults + penalty * next_eqs_eval,
                eq_mults_min,  # type: ignore
                eq_mults_max,  # type: ignore
            )

            next_sigma = torch.max(next_ineqs_eval, -ineq_mults / penalty)

            next_penalty = penalty
            if i > 0 and max([*torch.abs(next_eqs_eval), *torch.abs(next_sigma)]) > ratio * max(
                [*torch.abs(eqs_eval), *torch.abs(sigma)]  # type: ignore
            ):
                next_penalty *= penalty_growth

            p = p_next
            f_val = f(p)
            ineqs_eval = next_ineqs_eval
            eqs_eval = next_eqs_eval

            penalty = next_penalty
            sigma = next_sigma
            subsolver_tol = next_subsolver_tol
            ineq_mults = next_ineq_mults
            eq_mults = next_eq_mults

            # update values inside the float fields thereby changing the augmented lagrangian
            penalty_field.value = penalty
            for ineq_mult_field, ineq_mult in zip(ineq_mult_fields, ineq_mults):
                ineq_mult_field.value = ineq_mult
            for eq_mult_field, eq_mult in zip(eq_mult_fields, eq_mults):
                eq_mult_field.value = eq_mult

        if save_hist:
            f_hist.append(f_val.components.item())  # type: ignore
            ineqs_eval_hist.append(ineqs_eval)  # type: ignore
            eqs_eval_hist.append(eqs_eval)  # type: ignore
            p_hist.append(p.p)  # type: ignore

    if show_debug:
        print(f"[ralm] final p={p.p}, f={f_val.components}")

    # evaluate the constraints as scalar types
    ineqs_val = [ineq(p) for ineq in ineqs]
    eqs_val = [eq(p) for eq in eqs]

    result = ConstrResult(
        success=success,
        num_iters=i,
        p=p,
        f=f_val,
        ineqs=ineqs_val,
        eqs=eqs_val,
    )
    if save_hist:
        result.add_hist(f_hist, ineqs_hist, eqs_hist, p_hist)  # type: ignore
    return result
