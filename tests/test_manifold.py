import pytest
import torch

from torch.testing import assert_close

from dmol.diff_mfld.util import specs_match
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.vector_bundle import VectorBundle


class TestManifold:
    def test_type_gen(self):
        assert Manifold[0].dim == 0
        assert Manifold[1].dim == 1

        with pytest.raises(TypeError):
            Manifold[-1]

        with pytest.raises(TypeError):
            Manifold["dim"]

        with pytest.raises(TypeError):
            Manifold[0][0]

        with pytest.raises(TypeError):
            Manifold[0, 1]

        assert Manifold.incomplete
        assert not Manifold[2].incomplete

    def test_obj_gen(self):
        assert Manifold[0]("U").name == "U"

        with pytest.raises(ValueError):
            Manifold[0]("")

    def test_specs_match(self):
        M2 = Manifold[2]
        M3 = Manifold[2]

        # top-level manifolds are always considered distinct
        assert not specs_match(M2, M3)


class TestPoint:
    def test_type_gen(self):

        M1 = Manifold[1]
        M2 = Manifold[2]("U")  # instance

        assert Point[M1].manifold is M1
        with pytest.raises(TypeError):
            Point[M2]

        with pytest.raises(TypeError):
            Point[2]

        with pytest.raises(TypeError):
            Point[M2][M2]

        assert Point.incomplete
        assert not Point[M1].incomplete

        pass

    def test_obj_gen(self):
        M1 = Manifold[1]
        M2 = Manifold[2]

        # test with torch tensor

        assert_close(Point[M1](torch.tensor([0.0])).p, torch.tensor([0.0]))
        assert_close(Point[M2](torch.tensor([1.0, 2.0])).p, torch.tensor([1.0, 2.0]))

        with pytest.raises(TypeError):
            Point(torch.tensor([1.0, 2.0]))

        with pytest.raises(ValueError):
            Point[M2](torch.tensor([1.0]))

        # test feeding the point

        M3 = Manifold[2]  # same dimension

        p1 = Point[M1](torch.tensor([1.0]))
        p2 = Point[M2](torch.tensor([2.0, 3.0]))
        p3 = Point[M3](torch.tensor([2.0, 3.0]))

        with pytest.raises(ValueError):
            Point[M2](p1)

        assert Point[M2](p2)

        with pytest.raises(ValueError):
            Point[M2](p3)

    def test_spec_match(self):
        M = Manifold[2]

        # points are separate classes at runtime
        p1 = Point[M](torch.tensor([1.0, 2.0]))
        p2 = Point[M](torch.tensor([1.0, 2.0]))
        p3 = Point[M](torch.tensor([2.0, 3.0]))

        assert p1 == p2
        assert p1 != p3
        assert p2 != p3
