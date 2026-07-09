from dmol.diff_mfld.bundle.vector_bundle import TensorBundle, TangentBundle, TensorProductBundle, DualBundle
from dmol.diff_mfld.mfld import Manifold


class TestVectorBundle:
    def test_compatible_bundles(self):
        M = Manifold[2]

        assert TangentBundle[M].compatible_bundle(TangentBundle[M])
        assert TangentBundle[M].compatible_bundle(TensorBundle[1, 0][M])
        assert not TangentBundle[M].compatible_bundle(TensorBundle[1, 1][M])

        assert TensorProductBundle[
            DualBundle[TangentBundle[M]],
            DualBundle[TangentBundle[M]],
        ].compatible_bundle(TensorBundle[0, 2][M])
