import torch

from dmol.diff_mfld.bundle.vector_bundle import TangentBundle
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.field.field_types import LambdaField
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import MetricLambdaField, MetricField
from dmol.diff_mfld.testing import assert_tensors_equiv


class TestMetricField:
    def test_specs_match(self):
        M = Manifold[2]
        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [1.0 + x**2 * y**2, 0.0],  # type: ignore
                    [0.0, 1.0 + x**2 * y**2],  # type: ignore
                ],
            )
        )
        MetricField[M].validate_field(metric)

    def test_flat_field(self):
        M = Manifold[2]
        metric = MetricLambdaField[M](
            lambda x, y: coord_repr(
                [
                    [2.0, 0.0],  # type: ignore
                    [0.0, 2.0],
                ]
            )
        )
        vecf = LambdaField[TangentBundle[M]](lambda x, y: coord_repr([x, y]))  # type: ignore
        covf = metric.flat(vecf)  # type: ignore

        p = Point[M](torch.tensor([2.0, 3.0]))

        # expected
        vec = vecf(p)
        metric_tens = metric(p)
        cov_expected = metric_tens.flat(vec)

        # through the generated covector field
        cov = covf(p)

        assert_tensors_equiv(cov_expected, cov)
