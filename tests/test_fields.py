import torch

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
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


class TestAddFields:
    def test_scalar_add(self):
        pass
