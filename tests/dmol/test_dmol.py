import pytest
import torch
from torch.testing import assert_close

from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
from dmol.diff_mfld.connection.methods.geod_approx import approx_exp_map, approx_log_map
from dmol.diff_mfld.connection.methods.methods import Distance, DistanceMethod, ExpMapMethod, LogMapMethod
from dmol.diff_mfld.field.field_types import LambdaField
from dmol.diff_mfld.field.riem_fields import RiemSqrDist
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import EuclideanMetricField
from dmol.optim.constr.ralm import ralm
from dmol.torch.dmol import DiffMfldOptimLayer, DiffMfldOptimProblem

# the tests within this file are not meant to evaluate the performance of the layer but rather that the layer is able to
# execute and return a result without error both in batched and non-batched cases

# TODO: revise tests when/if methods are made more performant


class TestDMOL:
    @pytest.mark.parametrize("approx_order", [1])
    def test_approx_methods_unconstr_euclid(self, approx_order: int):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore

        retr = lambda p, v, conn: approx_exp_map(p, v, conn, approx_order)
        dist = lambda p, q, metric, conn: metric(p).norm(approx_log_map(p, q, conn, approx_order))
        log = lambda p, q, conn: approx_log_map(p, q, conn, approx_order)

        p0 = torch.tensor([1.0, 2.0])
        p0.requires_grad = True

        problem = DiffMfldOptimLayer(
            cost,  # type: ignore
            [],
            [],
            metric,
            conn=conn,
            retr=retr,
            dist=dist,
            log=log,
        )
        p_optim: torch.Tensor = problem(p0)
        p_optim.backward(gradient=torch.ones_like(p_optim))

        assert torch.norm(p0.grad) > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(p_optim).norm(conn.log(p_optim, torch.zeros((2,)))[0])
        assert pytest.approx(result_dist, abs=1e-2) == 0.0

    @pytest.mark.parametrize("approx_order", [1])
    def test_approx_methods_circ_constr_euclid(self, approx_order: int):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        constr_center = Point[M](torch.tensor([2.0, 3.0]))
        constr_radius = 0.75
        constr = RiemSqrDist[M](constr_center, metric, log_method=LogMapMethod.APPROX_O1) - constr_radius**2

        retr = lambda p, v, conn: approx_exp_map(p, v, conn, approx_order)
        dist = lambda p, q, metric, conn: metric(p).norm(approx_log_map(p, q, conn, approx_order))
        log = lambda p, q, conn: approx_log_map(p, q, conn, approx_order)

        p0 = torch.tensor([1.0, 2.0])
        p0.requires_grad = True

        problem = DiffMfldOptimLayer(
            cost,  # type: ignore
            (constr,),
            (),
            metric,
            conn=conn,
            retr=retr,
            dist=dist,
            log=log,
        )
        p_optim: torch.Tensor = problem(p0)
        p_optim.backward(gradient=torch.ones_like(p_optim))

        assert torch.norm(p0.grad) > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(p_optim).norm(conn.log(p_optim, constr_center)[0])
        assert pytest.approx(result_dist, abs=1e-2) == constr_radius
