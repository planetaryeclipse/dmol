import torch

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.vector_bundle import VectorBundle
from dmol.diff_mfld.bundle.tensor import Tensor


class TestTensor:
    def test_gen_type(self):
        M = Manifold[2]
        V = VectorBundle[2]

        T = Tensor[V][M]
        T(torch.tensor([1.0, 2.0]))
