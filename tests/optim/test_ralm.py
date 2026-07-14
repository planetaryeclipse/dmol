import pytest
import torch
from torch.testing import assert_close

from pytest import approx

from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
from dmol.diff_mfld.connection.methods.geod_approx import approx_exp_map, approx_log_map
from dmol.diff_mfld.connection.methods.methods import DistanceMethod, ExpMapMethod, LogMapMethod
from dmol.diff_mfld.field.field_types import LambdaField
from dmol.diff_mfld.field.riem_fields import RiemSqrDist
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import EuclideanMetricField, MetricLambdaField
from dmol.optim.constr.ralm import ralm
from dmol.optim.unconstr.rgd import rgd
from dmol.optim.unconstr.rtr import rtr

# TODO: revise these tests (with constrained approximation) at a later time when performance has been greatly improved


@pytest.mark.skip("current implementation is extremely slow so avoid running if possible")
class TestRalm:
    # unconstrained optimization (using subsolver)

    @pytest.mark.parametrize("subsolver_method", [rgd, rtr])
    def test_conn_methods_unconstr_euclid(self, subsolver_method):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        result = ralm(
            cost,  # type: ignore
            (),
            (),
            p0,
            metric,
            subsolver_method=subsolver_method,
        )
        assert result.success
        assert result.num_iters > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(result.p).norm(conn.log(result.p, torch.zeros((2,)))[0])
        assert approx(result_dist, abs=1e-2) == 0.0
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("approx_order", [1, 2, 3, 4])
    def test_approx_methods_unconstr_euclid(self, approx_order: int):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        retr = lambda p, v, conn: approx_exp_map(p, v, conn, approx_order)
        dist = lambda p, q, metric, conn: metric(p).norm(approx_log_map(p, q, conn, approx_order))
        result = ralm(
            cost,  # type: ignore
            (),
            (),
            p0,
            metric,
            retr=retr,
            dist=dist,
            subsolver_args={},
        )
        assert result.success
        assert result.num_iters > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(result.p).norm(conn.log(result.p, torch.zeros((2,)))[0])
        assert approx(result_dist, abs=1e-2) == 0.0
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    def test_conn_methods_unconstr_nonlinear(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ]
            )
        )
        conn = metric.levi_civita()

        result = ralm(
            cost,  # type: ignore
            (),
            (),
            p0,
            metric,
            subsolver_args={},
        )
        assert result.success
        assert result.num_iters > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(result.p).norm(conn.log(result.p, torch.zeros((2,)))[0])
        assert approx(result_dist, abs=1e-2) == 0.0
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("approx_order", [1, 2, 3, 4])
    def test_approx_methods_unconstr_nonlinear(self, approx_order: int):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ]
            )
        )
        conn = metric.levi_civita()

        result = ralm(
            cost,  # type: ignore
            (),
            (),
            p0,
            metric,
            subsolver_args={},
        )
        assert result.success
        assert result.num_iters > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(result.p).norm(conn.log(result.p, torch.zeros((2,)))[0])
        assert approx(result_dist, abs=1e-2) == 0.0
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    # using a circular region constraint

    def test_conn_methods_circ_constr_euclid(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        constr_center = Point[M](torch.tensor([2.0, 3.0]))
        constr_radius = 0.75
        constr = RiemSqrDist[M](constr_center, metric, log_method=LogMapMethod.APPROX_O1) - constr_radius**2

        p0 = Point[M](torch.tensor([1.0, 2.0]))
        result = ralm(
            cost,  # type: ignore
            (constr,),
            (),
            p0,
            metric,
            retr=ExpMapMethod.APPROX_O1,
            dist=DistanceMethod.APPROX_O1,
            subsolver_args={"damp": 0.1},
        )
        assert result.success
        assert result.num_iters > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(result.p).norm(conn.log(result.p, constr_center)[0])
        assert approx(result_dist, abs=1e-2) == constr_radius

    def test_conn_methods_circ_constr_nonlinear(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ]
            )
        )
        conn = metric.levi_civita()

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        constr_center = Point[M](torch.tensor([2.0, 3.0]))
        constr_radius = 0.75
        constr = RiemSqrDist[M](constr_center, metric) - constr_radius**2

        p0 = Point[M](torch.tensor([1.0, 2.0]))
        result = ralm(
            cost,  # type: ignore
            (constr,),
            (),
            p0,
            metric,
            subsolver_args={"damp": 0.1},
            penalty_start=0.1,
        )
        assert result.success
        assert result.num_iters > 0

        # uses riemann distance criterion so these errors are slightly larger
        result_dist = metric(result.p).norm(conn.log(result.p, constr_center)[0])
        assert approx(result_dist, abs=1e-2) == constr_radius
