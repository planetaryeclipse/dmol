import pytest

from dmol.diff_mfld.util import specs_match
from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.bundle.vector_bundle import (
    VectorBundle,
    DualBundle,
    ScalarBundle,
    TangentBundle,
    CotangentBundle,
    TensorProductBundle,
    KBundle,
    TensorBundle,
)


class TestVectorBundle:
    def test_type_gen(self):
        # test partial instantiation (only rank)

        V1 = VectorBundle[2]
        assert V1.incomplete
        assert V1.rank == 2
        assert V1.dim == 2
        assert V1.base is None

        V2 = VectorBundle[3]
        assert V2.incomplete
        assert V2.rank == 3
        assert V2.dim == 3
        assert V2.base is None

        # test min specialization case

        with pytest.raises(TypeError):
            VectorBundle[2, Manifold]

        B1 = VectorBundle
        with pytest.raises(TypeError):
            VectorBundle[2, B1]

        B2 = VectorBundle[2]
        V = VectorBundle[2, B2]
        assert V.incomplete

        # test full instantiation

        M = Manifold[2]
        V1 = VectorBundle[2, M]
        assert not V1.incomplete
        assert V1.rank == 2
        assert V1.dim == 2
        assert V1.base is M

        V2 = VectorBundle[3][M]
        assert not V2.incomplete
        assert V2.rank == 3
        assert V2.dim == 3
        assert V2.base is M

        # invalid rank parameter

        with pytest.raises(TypeError):
            VectorBundle[-1]
        with pytest.raises(TypeError):
            VectorBundle["dim"]

        # invalid base parameter

        with pytest.raises(TypeError):
            VectorBundle[1, None]
        with pytest.raises(TypeError):
            VectorBundle[1, 2]

        with pytest.raises(TypeError):
            VectorBundle[1][None]
        with pytest.raises(TypeError):
            VectorBundle[1][2]

        # test typing exhaustion

        with pytest.raises(TypeError):
            VectorBundle[3, M, 0]
        with pytest.raises(TypeError):
            VectorBundle[3, M][0]
        with pytest.raises(TypeError):
            VectorBundle[3][M][0]

        # test remaining specialization for bundles on bundles

        V3 = VectorBundle[4]
        V4 = VectorBundle[5]

        V = V3[V4]
        assert V.incomplete

        with pytest.raises(TypeError):
            V[0]
        with pytest.raises(TypeError):
            V["dim"]

        VM = V[M]
        assert not VM.incomplete
        assert not VM.base.incomplete
        assert VM.base.base is M

    def test_spec_match(self):
        M = Manifold[2]

        # defined separately so considered different classes at runtime
        V1 = VectorBundle[4][VectorBundle[5]][M]
        V2 = VectorBundle[4][VectorBundle[5]][M]

        V3 = VectorBundle[5][VectorBundle[5]][M]
        V4 = VectorBundle[4][VectorBundle[6]][M]

        assert specs_match(V1, V2)

        assert not specs_match(V1, V3)
        assert not specs_match(V1, V4)

    def test_obj_gen(self):
        with pytest.raises(TypeError):
            VectorBundle("vec")

        with pytest.raises(TypeError):
            VectorBundle[2]("vec")

        M = Manifold[2]
        VectorBundle[2, M]("dim")
        VectorBundle[2][M]("dim")

        with pytest.raises(ValueError):
            VectorBundle[2, M]("")
        with pytest.raises(ValueError):
            VectorBundle[2][M]("")


class TestDualBundle:
    def test_type_gen(self):
        # test min specialization case

        V = VectorBundle[4]
        D = DualBundle[V]
        assert D.incomplete
        assert D.dual is V

        # test full instantiation

        M = Manifold[2]

        V2 = VectorBundle[2, M]
        V4 = VectorBundle[4, M]

        DV2 = DualBundle[V2]
        assert not DV2.incomplete
        assert DV2.base is M
        assert DV2.rank == V2.rank
        assert DV2.dim == V2.dim
        assert DV2.dual is V2

        DV4 = DualBundle[V4]
        assert not DV4.incomplete
        assert DV4.base is M
        assert DV4.rank == V4.rank
        assert DV4.dim == V4.dim
        assert DV4.dual is V4

        # invalid bundle parameter

        with pytest.raises(TypeError):
            DualBundle[0]
        with pytest.raises(TypeError):
            DualBundle["dim"]
        with pytest.raises(TypeError):
            DualBundle[None]

        # test typing exhaustion

        with pytest.raises(TypeError):
            DualBundle[V][0]
        with pytest.raises(TypeError):
            DualBundle[V][M][0]

    def test_obj_gen(self):
        with pytest.raises(TypeError):
            DualBundle("dual")

        V1 = VectorBundle
        with pytest.raises(TypeError):
            DualBundle[V1]("dual")

        M = Manifold[2]
        V2 = VectorBundle[3, M]
        DualBundle[V2]("dual")

        with pytest.raises(ValueError):
            DualBundle[V2]("")


# TODO: flesh these tests out more (bare minimum to ensure instantiation for now)


class TestScalarBundle:
    def test_type_gen(self):
        M = Manifold[2]
        ScalarBundle[M]("dim")


class TestTangentBundle:
    def test_type_gen(self):
        M = Manifold[2]
        TangentBundle[M]("dim")


class TestCotangentBundle:
    def test_type_gen(self):
        M = Manifold[2]
        CotangentBundle[M]("dim")


class TestTensorProductBundle:
    def test_type_gen(self):
        M = Manifold[2]
        V1 = VectorBundle[2]
        V2 = VectorBundle[4]

        TPB = TensorProductBundle[V1, V2][M]
        TPB("dim")


class TestKBundle:
    def test_type_gen(self):
        M = Manifold[2]
        V = VectorBundle[2]

        KB = KBundle[3, V][M]
        KB("dim")


class TestTensorBundle:
    def test_type_gen(self):
        M = Manifold[2]

        TB = TensorBundle[1, 1][M]
        TB("dim")
