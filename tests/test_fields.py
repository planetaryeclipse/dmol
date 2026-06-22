from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
from dmol.diff_mfld.bundle.tensor import Tensor
from dmol.diff_mfld.bundle.field import LambdaField


class TestLambdaDiffField:
    def test_scalar(self):
        M = Manifold[2]
        S = LambdaField[Tensor[ScalarBundle]][M]
        s1 = S(lambda x, y: x * y)  # type: ignore
        s2 = S(lambda x, y: x + y)  # type: ignore


class TestAddFields:
    def test_scalar_add(self):
        pass
