import torch

from torch.testing import assert_close

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.vector_bundle import (
    ScalarBundle,
    VectorBundle,
    TangentBundle,
    KBundle,
    TensorBundle,
    TensorProductBundle,
)
from dmol.diff_mfld.bundle.tensor import Tensor
from dmol.diff_mfld.bundle.field import LambdaField


class TestLambdaDiffField:
    def test_scalar(self):
        M = Manifold[2]
        S = LambdaField[ScalarBundle][M]
        s1 = S(lambda x, y: x * y)  # type: ignore
        s2 = S(lambda x, y: x + y)  # type: ignore

        p = Point[M](torch.tensor([2.0, 3.0]))
        assert s1(p).components == 2.0 * 3.0
        assert s2(p).components == 2.0 + 3.0


class TestFieldOps:
    def test_add(self):
        M = Manifold[2]
        V = VectorBundle[3, M]

        v1 = LambdaField[V](lambda x, y: torch.tensor([y, -x, x * y]))  # type: ignore
        v2 = LambdaField[V](lambda x, y: torch.tensor([x * y, 1.0, 1.0]))  # type: ignore

        result_v1_v2 = v1 + v2

        p = Point[M](torch.tensor([1.0, 2.0]))
        assert_close(result_v1_v2(p).components, v1(p).components + v2(p).components)

    def test_mul(self):
        M = Manifold[2]
        V = VectorBundle[3, M]

        v = LambdaField[V](lambda x, y: torch.tensor([y, -x, x * y]))  # type: ignore

        # scalar bundle
        S = ScalarBundle[M]
        s = LambdaField[S](lambda x, y: 1.0 + x + y + x * y)  # type: ignore
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
