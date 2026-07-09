import pytest
import torch
from torch.testing import assert_close

from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
from dmol.diff_mfld.connection.methods.geod_approx import approx_exp_map
from dmol.diff_mfld.field.field_types import LambdaField
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import EuclideanMetricField, MetricLambdaField
from dmol.optim.unconstr.rtr import rtr


class TestRiemTrustRegion:
    def test_conn_methods_euclid(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        result = rtr(
            cost,  # type: ignore
            p0,
            metric,
            radius_max=0.15,
            radius_start=0.05,
            quality_step_thresh=0.15,
            tol=1e-4,
            max_iters=1000,
            retr=lambda p, v: conn.exp(p, v)[0],
            h=lambda v: v,
        )
        assert result.success
        assert result.num_iters > 0

        # unless radius_max is tightened then rtr usually has larger final error than rgd
        assert_close(result.p.p, torch.tensor([0.0, 0.0]), rtol=1e-1, atol=1e-1)  # type: ignore
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    def test_conn_methods_nonlinear(self):
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

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        result = rtr(
            cost,  # type: ignore
            p0,
            metric,
            radius_max=0.15,
            radius_start=0.05,
            quality_step_thresh=0.15,
            tol=1e-4,
            max_iters=1000,
            retr=lambda p, v: conn.exp(p, v)[0],
            h=lambda v: v,
        )
        assert result.success
        assert result.num_iters > 0

        assert_close(result.p.p, torch.tensor([0.0, 0.0]), rtol=1e-1, atol=1e-1)  # type: ignore
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("approx_order", [1, 2, 3, 4])
    def test_approx_methods_euclid(self, approx_order: int):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        result = rtr(
            cost,  # type: ignore
            p0,
            metric,
            radius_max=0.15,
            radius_start=0.05,
            quality_step_thresh=0.15,
            tol=1e-4,
            max_iters=1000,
            retr=lambda p, v: approx_exp_map(p, v, conn, approx_order=approx_order),
            h=lambda v: v,
        )
        assert result.success
        assert result.num_iters > 0

        assert_close(result.p.p, torch.tensor([0.0, 0.0]), rtol=1e-1, atol=1e-1)  # type: ignore
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("approx_order", [1, 2, 3, 4])
    def test_approx_methods_nonlinear(self, approx_order: int):
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

        cost = S(lambda x, y: coord_repr(1.0 + x**2 + y**2))  # type: ignore
        p0 = Point[M](torch.tensor([1.0, 2.0]))

        result = rtr(
            cost,  # type: ignore
            p0,
            metric,
            radius_max=0.15,
            radius_start=0.05,
            quality_step_thresh=0.15,
            tol=1e-4,
            max_iters=1000,
            retr=lambda p, v: approx_exp_map(p, v, conn, approx_order=approx_order),
            h=lambda v: v,
        )
        assert result.success
        assert result.num_iters > 0

        print(f"result f: {result.f.components}")

        assert_close(result.p.p, torch.tensor([0.0, 0.0]), rtol=1e-1, atol=1e-1)  # type: ignore
        assert_close(result.f.components, torch.tensor(1.0), rtol=1e-2, atol=1e-2)
