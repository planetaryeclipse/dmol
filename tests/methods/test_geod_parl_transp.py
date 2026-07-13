import pytest
import torch

from torch.testing import assert_close

from dmol.diff_mfld.connection.methods.geod_parl_transp import approx_parl_transp_vec
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.riemann import EuclideanMetricField, MetricLambdaField


class TestVector:

    def test_approx_euclidean(self):
        M = Manifold[2]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        p = Point[M](torch.tensor([2.0, 3.0]))
        v = Vec[M](torch.tensor([1.0, 1.0]))
        u = Vec[M](torch.tensor([-0.5, 0.5]))

        _, curve = conn.exp(p, v)
        w_ivp = conn.pt_vec(u, curve).components

        w_1 = approx_parl_transp_vec(u, p, v, conn, approx_order=1).components
        w_2 = approx_parl_transp_vec(u, p, v, conn, approx_order=2).components
        w_3 = approx_parl_transp_vec(u, p, v, conn, approx_order=3).components
        w_4 = approx_parl_transp_vec(u, p, v, conn, approx_order=4).components

        assert_close(w_ivp, w_1)
        assert_close(w_ivp, w_2)
        assert_close(w_ivp, w_3)
        assert_close(w_ivp, w_4)

    def test_approx_nonlinear(self):
        # no tests on accuracy but just ensure they can run for now

        M = Manifold[2]

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr([[1.0 + x**2 * y**2, 0.0], [0.0, 1.0 + x**2 * y**2]])  # type: ignore
        )
        conn = metric.levi_civita()

        p = Point[M](torch.tensor([2.0, 3.0]))

        v_tens = torch.tensor([1.0, 1.0])
        v = Vec[M](v_tens / torch.norm(v_tens) * 1.0)

        u_tens = torch.tensor([1.0, 0.0])
        u = Vec[M](u_tens)

        _, curve = conn.exp(p, v)
        w_ivp = conn.pt_vec(u, curve).components

        with pytest.raises(ValueError):
            approx_parl_transp_vec(u, p, v, conn, approx_order=0)

        w_1 = approx_parl_transp_vec(u, p, v, conn, approx_order=1).components
        w_2 = approx_parl_transp_vec(u, p, v, conn, approx_order=2).components
        w_3 = approx_parl_transp_vec(u, p, v, conn, approx_order=3).components
        w_4 = approx_parl_transp_vec(u, p, v, conn, approx_order=4).components

        err_norm_1 = torch.linalg.norm(w_ivp - w_1)
        err_norm_2 = torch.linalg.norm(w_ivp - w_2)
        err_norm_3 = torch.linalg.norm(w_ivp - w_3)
        err_norm_4 = torch.linalg.norm(w_ivp - w_4)

        print(f"w_ivp: {w_ivp}")
        print(f"w_1: {w_1}, err: {err_norm_1}")
        print(f"w_2: {w_2}, err: {err_norm_2}")
        print(f"w_3: {w_3}, err: {err_norm_3}")
        print(f"w_4: {w_4}, err: {err_norm_4}")

        assert err_norm_2 < err_norm_1
        assert err_norm_3 < err_norm_2
        assert err_norm_4 < err_norm_3

        # uncomment to ensure messages are shown
        # raise ValueError()
