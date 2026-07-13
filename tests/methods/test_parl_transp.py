import pytest
import torch
from torch.testing import assert_close

from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import EuclideanMetricField, MetricLambdaField


class TestVector:
    def test_ivp(self):
        M = Manifold[2]

        metric = EuclideanMetricField[M]()
        conn = metric.levi_civita()

        p = Point[M](torch.tensor([2.0, 3.0]))
        v = Vec[M](torch.tensor([1.0, 1.0]))
        u = Vec[M](torch.tensor([-0.5, 0.5]))

        # verify no change due to parallel transport in euclidean space
        _, curve = conn.exp(p, v)
        w = conn.pt_vec(u, curve)

        assert_close(w.components, u.components)  # unchanged in euclidean

        # verify parallel transport in nonlinear space
        metric_nonlinear = MetricLambdaField[M](
            lambda x, y: coord_repr([[1.0 + x**2 * y**2, 0.0], [0.0, 1.0 + x**2 * y**2]])  # type: ignore
        )
        conn_nonlinear = metric_nonlinear.levi_civita()

        _, curve_nonlinear = conn_nonlinear.exp(p, v)
        w_nonlinear = conn_nonlinear.pt_vec(u, curve_nonlinear)

        with pytest.raises(AssertionError):
            assert_close(w_nonlinear.components, w.components)
