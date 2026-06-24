import torch
import pytest

from torch.testing import assert_close

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.vector_bundle import (
    ScalarBundle,
    VectorBundle,
)
from dmol.diff_mfld.bundle.field import LambdaField, coord_repr, test_field_expr_callable_for_gradient
from dmol.diff_mfld.riemann import MetricLambdaField, EuclideanMetricField


class TestLambdaDiffField:
    def test_scalar(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]
        s1 = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s2 = S(lambda x, y: coord_repr(x + y))  # type: ignore

        p = Point[M](torch.tensor([2.0, 3.0]))
        assert s1(p).components == 2.0 * 3.0
        assert s2(p).components == 2.0 + 3.0


class TestFieldOps:
    def test_add(self):
        M = Manifold[2]
        V = VectorBundle[3, M]

        v1 = LambdaField[V](lambda x, y: coord_repr([y, -x, x * y]))  # type: ignore
        v2 = LambdaField[V](lambda x, y: coord_repr([x * y, 1.0, 1.0]))  # type: ignore

        result_v1_v2 = v1 + v2

        p = Point[M](torch.tensor([1.0, 2.0]))
        assert_close(result_v1_v2(p).components, v1(p).components + v2(p).components)

    def test_mul(self):
        M = Manifold[2]
        V = VectorBundle[3, M]

        v = LambdaField[V](lambda x, y: coord_repr([y, -x, x * y]))  # type: ignore

        # scalar bundle
        S = ScalarBundle[M]
        s = LambdaField[S](lambda x, y: coord_repr(1.0 + x + y + x * y))  # type: ignore
        result_v_s = v * s
        result_s_v = s * v
        p = Point[M](torch.tensor([1.0, 2.0]))
        assert_close(result_v_s(p).components, v(p).components * s(p).components)
        assert_close(result_s_v(p).components, s(p).components * v(p).components)

        # constant float
        s2 = 0.5
        result_v_s2 = v * s2
        result_s2_v = s2 * v

        assert_close(result_v_s2(p).components, v(p).components * s2)
        assert_close(result_s2_v(p).components, s2 * v(p).components)


class TestCovarDeriv:
    def test_scalar(self):
        M = Manifold[2]
        S = ScalarBundle[M]

        # euclidean space
        euclid_m = MetricLambdaField[M](lambda x, y: coord_repr([[1.0, 0.0], [0.0, 1.0]]))  # type: ignore
        euclid_conn = euclid_m.levi_civita()

        s = LambdaField[S](lambda x, y: coord_repr(1.0 + x**2 * y**2))  # type: ignore
        s_diff_euclid = euclid_conn.total_covar(s)
        s_hess_euclid = euclid_conn.total_covar(s_diff_euclid)

        x, y = 3.0, 4.0
        p = Point[M](torch.tensor([x, y]))
        assert_close(s(p).components, torch.tensor(1.0 + x**2 * y**2))
        assert_close(s_diff_euclid(p).components, torch.tensor([2 * x * y**2, 2 * x**2 * y]))
        assert_close(s_hess_euclid(p).components, torch.tensor([[2 * y**2, 4 * x * y], [4 * x * y, 2 * x**2]]))
        assert_close(euclid_conn.coeffs(p), torch.zeros((2, 2, 2)))

        # nonlinear metric
        nonlinear_m = MetricLambdaField[M](lambda x, y: coord_repr([[1.0 + x**2, 0.0], [0.0, 1.0 + y**2]]))  # type: ignore
        nonlinear_conn = nonlinear_m.levi_civita()

        s_diff_nonlinear = nonlinear_conn.total_covar(s)
        s_hess_nonlinear = nonlinear_conn.total_covar(s_diff_nonlinear)

        assert_close(s_diff_nonlinear(p).components, s_diff_euclid(p).components)  # differential matches
        with pytest.raises(AssertionError):  # hessian depends on connection coefficients
            assert_close(s_hess_nonlinear(p).components, s_hess_euclid(p).components)
        with pytest.raises(AssertionError):
            assert_close(nonlinear_conn.coeffs(p), torch.zeros((2, 2, 2)))

    def test_euclidean_metric_field(self):
        M = Manifold[2]
        euclid_m = EuclideanMetricField[M]()
        euclid_conn = euclid_m.levi_civita()

        p = Point[M](torch.tensor([3.0, 4.0]))
        assert_close(euclid_m(p).components, torch.eye(2))
        assert_close(euclid_conn.coeffs(p), torch.zeros((2, 2, 2)))

    def test_field_expr_checking(self):
        # using coord_repr preserves the gradient history
        test_field_expr_callable_for_gradient(
            lambda x, y: coord_repr([[1.0 + x**2, 0.0], [0.0, 1.0 + y**2]]),  # type: ignore
            coord_dim=2,
            single_arg=False,
            num_samples=20,
            coord_mean=2,
            coord_std=5,
        )

        # using torch.tensor() directly breaks the gradient history (use this in implementation-specific tests)
        with pytest.raises(ValueError):
            test_field_expr_callable_for_gradient(
                lambda x, y: torch.tensor([[1.0 + x**2, 0.0], [0.0, 1.0 + y**2]]),  # type: ignore
                coord_dim=2,
                single_arg=False,
                num_samples=20,
                coord_mean=2,
                coord_std=5,
            )
