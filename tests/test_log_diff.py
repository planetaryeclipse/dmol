import pytest
import torch

from torch.testing import assert_close

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.riemann import EuclideanMetricField, MetricLambdaField

from dmol.diff_mfld.connection.methods.geod_ivp_bvp import bvp_log_map
from dmol.diff_mfld.connection.methods.log_diff import approx_log_covar
from dmol.diff_mfld.connection.methods.geod_approx import approx_log_map


class TestLogdiff:
    def test_approx_euclidean(self):
        M = Manifold[2]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        p = Point[M](torch.tensor([2.0, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        v_bvp = bvp_log_map(p, q, conn)[0]

        log_covar_1 = approx_log_covar(p, q, v_bvp, conn, approx_order=1).components
        log_covar_2 = approx_log_covar(p, q, v_bvp, conn, approx_order=2).components
        log_covar_3 = approx_log_covar(p, q, v_bvp, conn, approx_order=3).components
        log_covar_4 = approx_log_covar(p, q, v_bvp, conn, approx_order=4).components

        v_expected = -torch.eye(2)  # for euclidean case
        assert_close(v_expected, log_covar_1)
        assert_close(v_expected, log_covar_2)
        assert_close(v_expected, log_covar_3)
        assert_close(v_expected, log_covar_4)

    def test_approx_nonlinear(self):
        # no tests on accuracy but just ensure that they can run for now

        M = Manifold[2]

        # p = torch.tensor([2.0, 1.0])
        p = torch.tensor([2.0, 3.0])  # original q
        q = torch.tensor([2.5, 0.5])

        metric_nonlinear = MetricLambdaField[M](
            lambda x, y: coord_repr([[1.0 + x**2 * y**2, 0.0], [0.0, 1.0 + x**2 * y**2]])  # type: ignore
        )
        conn_nonlinear = metric_nonlinear.levi_civita()

        v, _ = bvp_log_map(p, q, conn_nonlinear)

        with pytest.raises(ValueError):
            approx_log_covar(p, q, v, conn_nonlinear, approx_order=0)

        v_partials_1 = approx_log_covar(p, q, v, conn_nonlinear, approx_order=1).components
        v_partials_2 = approx_log_covar(p, q, v, conn_nonlinear, approx_order=2).components
        v_partials_3 = approx_log_covar(p, q, v, conn_nonlinear, approx_order=3).components
        v_partials_4 = approx_log_covar(p, q, v, conn_nonlinear, approx_order=4).components

        # obtained from taking high-accuracy finite difference
        # expected = torch.tensor([[-0.6069, -0.3983], [0.4746, -0.6613]])  #  + torch.einsum(
        #     "k,ijk->ij", v.components, conn_nonlinear.coeffs(p)
        # )

        # with original q
        expected = torch.tensor([[0.4716, -0.7853], [0.0268, 0.0047]]) + torch.einsum(
            "k,ijk->ij", v.components, conn_nonlinear.coeffs(p)
        )

        err_norm_1 = torch.linalg.matrix_norm(expected - v_partials_1)
        err_norm_2 = torch.linalg.matrix_norm(expected - v_partials_2)
        err_norm_3 = torch.linalg.matrix_norm(expected - v_partials_3)
        err_norm_4 = torch.linalg.matrix_norm(expected - v_partials_4)

        print(f"bvp v: {v}")
        print(f"expected v_partials: {expected}")
        print(f"v_partials_1: {v_partials_1}, err: {err_norm_1}")
        print(f"v_partials_2: {v_partials_2}, err: {err_norm_2}")
        print(f"v_partials_3: {v_partials_3}, err: {err_norm_3}")
        print(f"v_partials_4: {v_partials_4}, err: {err_norm_4}")

        assert err_norm_2 < err_norm_1
        assert err_norm_3 < err_norm_2
        assert err_norm_4 < err_norm_3

        # uncomment to ensure messages are shown
        # raise ValueError()
