import torch

from dmol.diff_mfld.connection.methods.geod_ivp_bvp import bvp_log_map
from dmol.diff_mfld.connection.methods.geod_log_diff import approx_log_covar
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import MetricLambdaField

from dmol.diff_mfld.connection.methods.methods import LogMapMethod, LogMapCovarMethod
from dmol.diff_mfld.field.riem_fields import RiemSqrDist, RiemLog
from dmol.diff_mfld.testing import assert_tensors_equiv


class TestRiemSqrDist:
    def test_bvp(self):
        M = Manifold[2]

        p = Point[M](torch.tensor([2.0, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ],
            )
        )
        conn = metric.levi_civita()

        # computes the distance manually
        v = bvp_log_map(p, q, conn)[0]
        expected_dist_sqr = metric(p).inner(v, v)

        # compute using the field
        riem_sqr_field = RiemSqrDist[M](q, metric, log_method=LogMapMethod.BVP)
        dist_sqr = riem_sqr_field(p)

        assert_tensors_equiv(expected_dist_sqr, dist_sqr)

    def test_bvp_covar(self):
        M = Manifold[2]

        p = Point[M](torch.tensor([2.0, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ],
            )
        )
        conn = metric.levi_civita()

        # computes the differential manually
        v = -2.0 * bvp_log_map(p, q, conn)[0]
        cov_expected = metric(p).flat(v)

        # computes automatically
        riem_sqr_field = RiemSqrDist[M](q, metric, log_method=LogMapMethod.BVP)
        riem_sqr_field_covar = conn.total_covar(riem_sqr_field)
        cov = riem_sqr_field_covar(p)

        assert_tensors_equiv(cov_expected, cov)

    def test_bvp_covar_hess(self):
        M = Manifold[2]

        p = Point[M](torch.tensor([2.0, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ],
            )
        )
        conn = metric.levi_civita()

        # computes the hessian manually manually
        v = bvp_log_map(p, q, conn)[0]
        v_covar = approx_log_covar(p, q, v, conn, approx_order=4)
        hess_expected = metric(p).flat(-2.0 * v_covar)

        # computes automatically
        riem_sqr_field = RiemSqrDist[M](
            q,
            metric,
            log_method=LogMapMethod.BVP,
            log_covar_method=LogMapCovarMethod.APPROX_O4,
        )
        riem_sqr_field_covar = conn.total_covar(riem_sqr_field)
        riem_sqr_field_hess = conn.total_covar(riem_sqr_field_covar)
        print(f"after making riem sqr field hess: {riem_sqr_field_hess}")
        print()
        hess = riem_sqr_field_hess(p)

        assert_tensors_equiv(hess_expected, hess)


class TestRiemLog:
    def test_bvp(self):
        M = Manifold[2]

        p = Point[M](torch.tensor([2.0, 3.0]))
        q = Point[M](torch.tensor([2.5, 0.5]))

        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ],
            )
        )
        conn = metric.levi_civita()

        # computes the riemannian log manually
        v_expected = bvp_log_map(p, q, conn)[0]

        # computes using the field
        riem_log_field = RiemLog[M](q, metric, log_method=LogMapMethod.BVP)
        v = riem_log_field(p)

        assert_tensors_equiv(v_expected, v)
