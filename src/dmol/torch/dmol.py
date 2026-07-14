from typing import Any, Sequence, Callable

import torch

from torch.autograd.function import Function
from torch.nn import Module

from dmol.diff_mfld.bundle.tensor import Tensor, Vec
from dmol.diff_mfld.bundle.vector_bundle import TensorBundle
from dmol.diff_mfld.connection.methods.methods import (
    Distance,
    DistanceMethod,
    ExpMapMethod,
    GeodParlTransp,
    GeodParlTranspMethod,
    Log,
    LogMapMethod,
)
from dmol.diff_mfld.connection.tangent import TangentConnection
from dmol.diff_mfld.field.field_types import ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import MetricField
from dmol.optim.constr.ralm import ralm
from dmol.optim.methods import ConstrOptimFn, Retraction


def compute_soln_map_jacob[M: Manifold](
    p0: Point[M] | torch.Tensor,
    v0: Vec[M],
    p_optimal: Point[M] | torch.Tensor,
    ineq_mults: torch.Tensor,
    f: ScalarField[M],
    ineqs: list[ScalarField[M]],
    metric: MetricField[M],
    conn: TangentConnection[M],
    geod_parl_transp: GeodParlTransp[M],
) -> Tensor[TensorBundle[1, 1]]:
    p0 = Point[f.bundle.base](p0)
    p_optimal = Point[f.bundle.base](p_optimal)

    f_diff = conn.total_covar(f)
    f_hess = conn.total_covar(f_diff)
    ineqs_hess = [conn.total_covar(conn.total_covar(ineq)) for ineq in ineqs]

    # computes the kkt dual (holds at the optimal point)
    kkt_map_dual = -f_hess(p_optimal).components
    for i in range(len(ineqs)):
        ineq_mult = ineq_mults[i]
        ineq_value = ineqs[i](p_optimal).components.item()
        ineq_partials = ineqs[i].partials(p_optimal)
        ineq_hess = ineqs_hess[i](p_optimal).components

        kkt_map_dual += ineq_mult / ineq_value * torch.outer(ineq_partials, ineq_partials)
        kkt_map_dual -= ineq_mult * ineq_hess

    # computes the kkt endomorphism (exists at the optimal point)
    metric_mat = metric(p_optimal).components
    kkt_endo = metric_mat @ kkt_map_dual

    # computes the transition matrices for parallel transport
    n = f.bundle.base.dim
    parl_transp_trans_mat = torch.zeros((n, n))
    for i in range(n):
        basis = torch.zeros((n,))
        basis[i] = 1.0
        basis_vec = Vec[f.bundle.base](basis)

        transp_basis_vec = geod_parl_transp(basis_vec, p0, v0, conn)
        parl_transp_trans_mat[:, i] = transp_basis_vec.components

    # computes the solution map jacobian
    soln_jacob = kkt_endo @ parl_transp_trans_mat

    return Tensor[TensorBundle[1, 1][f.bundle.base]](soln_jacob)


class DiffMfldOptimProblem[M: Manifold](Function):
    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,
        f: ScalarField[M],
        ineqs: Sequence[ScalarField[M]],
        eqs: Sequence[ScalarField[M]],
        metric: MetricField[M],
        solver_method: ConstrOptimFn[M],
        solver_args: dict[str, Any],
        conn: TangentConnection[M] | None,
        retr: Retraction[M],
        dist: Distance[M] | None,
        log: Log[M],
        geod_parl_transp: GeodParlTransp[M],
        tol: float,
        max_iters: int,
    ):
        if conn is None:
            conn = metric.levi_civita()
        result = solver_method(
            f,
            ineqs,
            eqs,
            Point[f.bundle.base](p0),
            metric,
            conn,
            retr,
            dist,
            tol,
            max_iters,
            False,  # no history
            False,  # no debug messages
            **solver_args,
        )
        if not result.success:
            raise ValueError("manifold optimization failed")
        if result.ineq_mults is None:
            raise ValueError("constrained solver must generate lagrangian multipliers for inequality constraints")

        p_optimal = result.p.p  # type: ignore
        v0 = log(p0, p_optimal, conn).components

        ctx.save_for_backward(p0, v0, p_optimal)
        ctx.f = f
        ctx.ineqs = ineqs
        ctx.ineq_mults = result.ineq_mults
        ctx.metric = metric
        ctx.conn = conn
        ctx.geod_parl_transp = geod_parl_transp

        return p_optimal

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        p0, v0, p_optimal = ctx.saved_tensors

        f: ScalarField[M]
        ineqs: list[ScalarField[M]]
        ineq_mults: torch.Tensor
        metric: MetricField[M]
        conn: TangentConnection[M]
        geod_parl_transp: GeodParlTransp[M]

        f, ineqs, ineq_mults, metric, conn, geod_parl_transp = (
            ctx.f,
            ctx.ineqs,
            ctx.ineq_mults,
            ctx.metric,
            ctx.conn,
            ctx.geod_parl_transp,
        )
        soln_jacob = compute_soln_map_jacob(
            p0,
            Vec[f.bundle.base](v0),
            p_optimal,
            ineq_mults,
            f,
            ineqs,
            metric,
            conn,
            geod_parl_transp,
        )
        soln_jacob_mat = soln_jacob.components
        backprop_grad_vec_p0 = torch.linalg.solve(soln_jacob_mat, grad_output)

        # gradient only flows through the tangent space at the original point
        return backprop_grad_vec_p0, *(None for _ in range(13))


class DiffMfldOptimLayer[M: Manifold](Module):
    def __init__(
        self,
        f: ScalarField[M],
        ineqs: Sequence[ScalarField[M]],
        eqs: Sequence[ScalarField[M]],
        metric: MetricField[M],
        solver_method: ConstrOptimFn[M] = ralm,
        solver_args: dict[str, Any] = {},
        *,
        conn: TangentConnection[M] | None = None,
        retr: Retraction[M] = ExpMapMethod.DEFAULT,
        dist: Distance[M] = DistanceMethod.DEFAULT,
        log: Log[M] = LogMapMethod.DEFAULT,
        geod_parl_transp: GeodParlTransp[M] = GeodParlTranspMethod.DEFAULT,
        tol: float = 1e-3,
        max_iters: int = 1000,
    ):
        super().__init__()
        self._f = f
        self._ineqs = ineqs
        self._eqs = eqs
        self._metric = metric
        self._solver_method = solver_method
        self._solver_args = solver_args

        if conn is None:
            conn = metric.levi_civita()

        self._conn = conn
        self._retr = retr
        self._dist = dist
        self._log = log
        self._geod_parl_transp = geod_parl_transp
        self._tol = tol
        self._max_iters = max_iters

    def forward(
        self,
        p0: torch.Tensor,
    ):
        print(f"p0 shape: {p0.shape}, dim: {p0.dim()}")

        is_batched = p0.dim() > 1
        if is_batched:
            p0_batched = p0
            p_optimal_batched = torch.zeros_like(p0_batched)

            num_batches = p0_batched.shape[0]
            for i in range(num_batches):
                p_optimal_batched[i, :] = DiffMfldOptimProblem.apply(
                    p0_batched[i, :],
                    self._f,
                    self._ineqs,
                    self._eqs,
                    self._metric,
                    self._solver_method,
                    self._solver_args,
                    self._conn,
                    self._retr,
                    self._dist,
                    self._log,
                    self._geod_parl_transp,
                    self._tol,
                    self._max_iters,
                )
            return p_optimal_batched
        else:
            return DiffMfldOptimProblem.apply(
                p0,
                self._f,
                self._ineqs,
                self._eqs,
                self._metric,
                self._solver_method,
                self._solver_args,
                self._conn,
                self._retr,
                self._dist,
                self._log,
                self._geod_parl_transp,
                self._tol,
                self._max_iters,
            )
