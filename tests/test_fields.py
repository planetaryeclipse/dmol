import torch
import pytest

from torch.testing import assert_close

from dmol.diff_mfld.bundle.tensor import Cov, Scalar, Tensor
from dmol.diff_mfld.field.field_types import CovectorField, FloatField, LambdaField, ScalarField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.vector_bundle import (
    ScalarBundle,
    TensorBundle,
    VectorBundle,
)

from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.field.testing import check_field_expr_callable_for_gradient
from dmol.diff_mfld.riemann import MetricLambdaField, EuclideanMetricField
from dmol.diff_mfld.testing import assert_tensors_equiv


class TestLambdaField:
    def test_scalar(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]
        s1 = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s2 = S(lambda x, y: coord_repr(x + y))  # type: ignore

        p = Point[M](torch.tensor([2.0, 3.0]))
        assert s1(p).components == 2.0 * 3.0
        assert s2(p).components == 2.0 + 3.0


class TestFloatField:
    def test_float(self):
        M = Manifold[2]

        f1 = FloatField[M](2.0)
        f2 = FloatField[M](3.0)

        p = Point[M](torch.tensor([2.0, 3.0]))
        assert_tensors_equiv(f1(p), torch.tensor(2.0))
        assert_tensors_equiv(f2(p), torch.tensor(3.0))

    def test_float_covar(self):
        M = Manifold[2]

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ],
            )
        )
        conn = metric.levi_civita()

        f1_covar = conn.total_covar(FloatField[M](2.0))
        f2_covar = conn.total_covar(FloatField[M](3.0))

        p = Point[M](torch.tensor([2.0, 3.0]))
        assert_tensors_equiv(f1_covar(p), Cov[M](torch.zeros((2,))))
        assert_tensors_equiv(f2_covar(p), Cov[M](torch.zeros((2,))))

    def test_change_value(self):
        M = Manifold[2]
        f = FloatField[M](2.0)

        p = Point[M](torch.tensor([2.0, 3.0]))
        assert_tensors_equiv(f(p), 2.0)

        f.value = 3.0
        assert_tensors_equiv(f(p), 3.0)

    def test_change_under_composition(self):
        M = Manifold[2]
        f = FloatField[M](2.0)
        f_pow = f**2

        p = Point[M](torch.tensor([2.0, 3.0]))
        assert_tensors_equiv(f_pow(p), 2.0**2)

        f.value = 3.0
        assert_tensors_equiv(f_pow(p), 3.0**2)


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

    def test_div(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle[M]]
        s1 = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s2 = S(lambda x, y: coord_repr(x + y))  # type: ignore

        s_div_1 = s1 / s2
        s_div_2 = s2 / s1

        p = Point[M](torch.tensor([1.0, 2.0]))

        assert_tensors_equiv(s_div_1(p), 2.0 / 3.0)
        assert_tensors_equiv(s_div_2(p), 3.0 / 2.0)

    def test_div_float(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle[M]]
        s = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s_div = 1.0 / s

        p = Point[M](torch.tensor([1.0, 2.0]))
        assert_tensors_equiv(s_div(p), 0.5)

    def test_max(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle[M]]
        s1 = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s2 = S(lambda x, y: coord_repr(x + y))  # type: ignore

        max_s = ScalarField.max(s1, s2)

        p1 = Point[M](torch.tensor([0.5, 0.5]))
        p2 = Point[M](torch.tensor([2.0, 2.0]))

        assert_tensors_equiv(max_s(p1), 1.0)  # add
        assert_tensors_equiv(max_s(p2), 4.0)  # multiply

    def test_power(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle[M]]
        s = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s_pow = s**2

        p = Point[M](torch.tensor([1.0, 2.0]))
        assert_tensors_equiv(s_pow(p), 4.0)


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

    def test_scalar_mul(self):
        M = Manifold[2]
        V = VectorBundle[3, M]
        S = ScalarBundle[M]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        f = LambdaField[S](lambda x, y: coord_repr(x**2 * y**2))  # type: ignore
        g = LambdaField[S](lambda x, y: coord_repr(x**2 + y**2))  # type: ignore

        mult_1 = f * g
        mult_2 = g * f

        assert ScalarField[M].compatible_field(mult_1)
        assert ScalarField[M].compatible_field(mult_2)

        x, y = 1.0, 2.0
        p = Point[M](torch.tensor([x, y]))

        mult_1_diff = conn.total_covar(mult_1)
        mult_2_diff = conn.total_covar(mult_2)

        assert CovectorField[M].compatible_field(mult_1_diff)
        assert CovectorField[M].compatible_field(mult_2_diff)

        f_val = x**2 * y**2
        g_val = x**2 + y**2

        f_diff_val = torch.tensor([2 * x * y**2, 2 * x**2 * y])
        g_diff_val = torch.tensor([2 * x, 2 * y])

        f_hess_val = torch.tensor(
            [
                [2 * y**2, 4 * x * y],
                [4 * x * y, 2 * x**2],
            ]
        )
        g_hess_val = 2 * torch.eye(2)

        mult_diff_tens_val = f_diff_val * g_val + f_val * g_diff_val
        assert_tensors_equiv(
            mult_1_diff(p),
            Cov[M](mult_diff_tens_val),
        )
        assert_tensors_equiv(
            mult_2_diff(p),
            Cov[M](mult_diff_tens_val),
        )

        mult_1_hess = conn.total_covar(mult_1_diff)
        mult_2_hess = conn.total_covar(mult_2_diff)

        mult_hess_tens_val = (
            f_hess_val * g_val
            + torch.outer(f_diff_val, g_diff_val)
            + torch.outer(g_diff_val, f_diff_val)
            + f_val * g_hess_val
        )
        assert_tensors_equiv(
            mult_1_hess(p),
            Tensor[TensorBundle[0, 2]][M](mult_hess_tens_val),
        )
        assert_tensors_equiv(
            mult_2_hess(p),
            Tensor[TensorBundle[0, 2]][M](mult_hess_tens_val),
        )

    def test_euclidean_metric_field(self):
        M = Manifold[2]
        euclid_m = EuclideanMetricField[M]()
        euclid_conn = euclid_m.levi_civita()

        p = Point[M](torch.tensor([3.0, 4.0]))
        assert_close(euclid_m(p).components, torch.eye(2))
        assert_close(euclid_conn.coeffs(p), torch.zeros((2, 2, 2)))

    @pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True")
    def test_field_expr_checking(self):
        # using coord_repr preserves the gradient history
        check_field_expr_callable_for_gradient(
            lambda x, y: coord_repr([[1.0 + x**2, 0.0], [0.0, 1.0 + y**2]]),  # type: ignore
            coord_dim=2,
            single_arg=False,
            num_samples=20,
            coord_mean=2,
            coord_std=5,
        )

        # using torch.tensor() directly breaks the gradient history (use this in implementation-specific tests)
        with pytest.raises(ValueError):
            check_field_expr_callable_for_gradient(
                lambda x, y: torch.tensor([[1.0 + x**2, 0.0], [0.0, 1.0 + y**2]]),  # type: ignore
                coord_dim=2,
                single_arg=False,
                num_samples=20,
                coord_mean=2,
                coord_std=5,
            )

    def test_div_covar_nonlinear(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle[M]]
        s1 = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s2 = S(lambda x, y: coord_repr(x + y))  # type: ignore

        s_div_1 = s1 / s2
        s_div_2 = s2 / s1

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2, 0.0],  # type: ignore
                    [0.0, 1.0 + y**2],  # type: ignore
                ]
            )
        )
        conn = metric.levi_civita()
        s1_covar = conn.total_covar(s1)
        s2_covar = conn.total_covar(s2)
        s_div_1_covar = conn.total_covar(s_div_1)
        s_div_2_covar = conn.total_covar(s_div_2)

        p = Point[M](torch.tensor([1.0, 2.0]))
        assert_tensors_equiv(s_div_1_covar(p), 1.0 / s2(p) * s1_covar(p) + s1(p) / s2(p) ** 2 * s2_covar(p))
        assert_tensors_equiv(s_div_2_covar(p), 1.0 / s1(p) * s2_covar(p) + s2(p) / s1(p) ** 2 * s1_covar(p))

    def test_max_covar_nonlinear(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle[M]]
        s1 = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s2 = S(lambda x, y: coord_repr(x + y))  # type: ignore
        max_s = ScalarField.max(s1, s2)

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2, 0.0],  # type: ignore
                    [0.0, 1.0 + y**2],  # type: ignore
                ]
            )
        )
        conn = metric.levi_civita()
        max_s_covar = conn.total_covar(max_s)

        p1 = Point[M](torch.tensor([0.5, 0.5]))
        p2 = Point[M](torch.tensor([2.0, 2.0]))

        assert_tensors_equiv(max_s_covar(p1), Cov[M](torch.tensor([1.0, 1.0])))  # add
        assert_tensors_equiv(max_s_covar(p2), Cov[M](torch.tensor([2.0, 2.0])))  # multiply

    def test_power_covar_nonlinear(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle[M]]
        s = S(lambda x, y: coord_repr(x * y))  # type: ignore
        s_pow = s**2

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2, 0.0],  # type: ignore
                    [0.0, 1.0 + y**2],  # type: ignore
                ]
            )
        )
        conn = metric.levi_civita()
        s_pow_covar = conn.total_covar(s_pow)

        p = Point[M](torch.tensor([1.0, 2.0]))
        print(s_pow_covar(p).components)
        assert_tensors_equiv(s_pow_covar(p), 2.0 * 2.0 * Cov[M](torch.tensor([2.0, 1.0])))
