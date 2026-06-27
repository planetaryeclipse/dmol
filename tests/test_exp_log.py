import pytest
import torch

from torch.testing import assert_close

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.bundle.field import coord_repr
from dmol.diff_mfld.riemann import EuclideanMetricField, MetricLambdaField

from dmol.diff_mfld.connection.geod_approx import approx_exp_map, approx_log_map


class TestExpMap:
    def test_ivp(self):
        M = Manifold[2]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        # verify the geodesic in euclidean space
        p0_tens = torch.tensor([0.0, 0.0])
        v0_tens = torch.tensor([1.0, 2.0])

        p = Point[M](p0_tens)
        v = Vec[M](v0_tens)
        q, curve = conn.exp(p, v)

        assert_close(q.p, p0_tens + v0_tens)

        # verify the output curve
        p0_curve, v0_curve = curve.sample(0.0)
        pf_curve, vf_curve = curve.sample(1.0)
        p_curve_initial, v_curve_initial = curve.initial
        min_time_curve, max_time_curve = curve.interval

        assert_close(p0_curve.p, p0_tens)
        assert_close(v0_curve.components, v0_tens)
        assert_close(pf_curve.p, p0_tens + v0_tens)
        assert_close(vf_curve.components, v0_tens)  # given euclidean
        assert_close(p_curve_initial.p, p0_tens)
        assert_close(v_curve_initial.components, v0_tens)
        assert_close(min_time_curve, 0.0)
        assert_close(max_time_curve, 1.0)

        # verify the geodesic in nonlinear space
        metric_nonlinear = MetricLambdaField[M](
            lambda x, y: coord_repr([[1.0 + x**2 * y**2, 0.0], [0.0, 1.0 + x**2 * y**2]])  # type: ignore
        )
        conn_nonlinear = metric_nonlinear.levi_civita()

        q_nonlinear, _ = conn_nonlinear.exp(p, v)
        with pytest.raises(AssertionError):
            assert_close(q_nonlinear.p, p0_tens + v0_tens)

    def test_approx_euclid(self):
        M = Manifold[2]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        p = Point[M](torch.tensor([0.0, 0.0]))
        v = Vec[M](torch.tensor([1.0, 2.0]))

        q_ivp, _ = conn.exp(p, v)

        q_approx = approx_exp_map(p, v, conn, approx_order=1)
        assert_close(q_approx.p, q_ivp.p)

        q_approx = approx_exp_map(p, v, conn, approx_order=2)
        assert_close(q_approx.p, q_ivp.p)

        q_approx = approx_exp_map(p, v, conn, approx_order=3)
        assert_close(q_approx.p, q_ivp.p)

        q_approx = approx_exp_map(p, v, conn, approx_order=4)
        assert_close(q_approx.p, q_ivp.p)

    def test_approx_nonlinear(self):
        # no tests on accuracy but just ensure that they can run for now

        M = Manifold[2]

        p = Point[M](torch.tensor([0.0, 0.0]))
        v = Vec[M](torch.tensor([1.0, 2.0]))

        metric_nonlinear = MetricLambdaField[M](
            lambda x, y: coord_repr([[1.0 + x**2 * y**2, 0.0], [0.0, 1.0 + x**2 * y**2]])  # type: ignore
        )
        conn_nonlinear = metric_nonlinear.levi_civita()

        p = Point[M](torch.tensor([0.0, 0.0]))
        v = Vec[M](torch.tensor([1.0, 2.0]))

        with pytest.raises(ValueError):
            approx_exp_map(p, v, conn_nonlinear, 0)

        approx_exp_map(p, v, conn_nonlinear, 1)
        approx_exp_map(p, v, conn_nonlinear, 2)
        approx_exp_map(p, v, conn_nonlinear, 3)
        approx_exp_map(p, v, conn_nonlinear, 4)


class TestLogMap:
    def test_bvp(self):
        M = Manifold[2]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        p = Point[M](torch.tensor([-1.5, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        v, curve = conn.log(p, q)
        q_again, _ = conn.exp(p, v)

        assert_close(q.p, q_again.p, rtol=1e-3, atol=1e-3)
        p0_curve, v0_curve = curve.sample(0.0)
        pf_curve, _ = curve.sample(1.0)
        assert_close(
            p0_curve.p,
            p.p,
        )
        assert_close(v0_curve.components, v.components)
        assert_close(pf_curve.p, q.p, rtol=1e-3, atol=1e-3)

        # nonlinear test
        metric_nonlinear = MetricLambdaField[M](
            lambda x, y: coord_repr([[1.0 + x**2 * y**2, 0.0], [0.0, 1.0 + x**2 * y**2]])  # type: ignore
        )
        conn_nonlinear = metric_nonlinear.levi_civita()

        v, _ = conn_nonlinear.log(p, q)
        q_again, _ = conn_nonlinear.exp(p, v)
        assert_close(q.p, q_again.p, rtol=1e-3, atol=1e-3)

        # inconsistent connection usage
        v, _ = conn.log(p, q)
        q_again, _ = conn_nonlinear.exp(p, v)
        with pytest.raises(AssertionError):
            assert_close(q.p, q_again.p, rtol=1e-3, atol=1e-3)

    def test_approx_euclidean(self):
        M = Manifold[2]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        p = Point[M](torch.tensor([-1.5, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        v_bvp, _ = conn.log(p, q)

        v_approx = approx_log_map(p, q, conn, approx_order=1)
        assert_close(v_approx.components, v_bvp.components)

        v_approx = approx_log_map(p, q, conn, approx_order=2)
        assert_close(v_approx.components, v_bvp.components)

        v_approx = approx_log_map(p, q, conn, approx_order=3)
        assert_close(v_approx.components, v_bvp.components)

        v_approx = approx_log_map(p, q, conn, approx_order=4)
        assert_close(v_approx.components, v_bvp.components)

    def test_approx_nonlinear(self):
        # no tests on accuracy but just ensure that they can run for now

        M = Manifold[2]

        metric_nonlinear = MetricLambdaField[M](
            lambda x, y: coord_repr([[1.0 + x**2 * y**2, 0.0], [0.0, 1.0 + x**2 * y**2]])  # type: ignore
        )
        conn_nonlinear = metric_nonlinear.levi_civita()

        p = Point[M](torch.tensor([-1.5, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        with pytest.raises(ValueError):
            approx_log_map(p, q, conn_nonlinear, approx_order=0)

        approx_log_map(p, q, conn_nonlinear, approx_order=1)
        approx_log_map(p, q, conn_nonlinear, approx_order=2)
        approx_log_map(p, q, conn_nonlinear, approx_order=3)
        approx_log_map(p, q, conn_nonlinear, approx_order=4)
