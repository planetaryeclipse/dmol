import torch
import pytest

from torch.testing import assert_close

from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.bundle.vector_bundle import VectorBundle, TangentBundle, TensorProductBundle, TensorBundle, KBundle
from dmol.diff_mfld.bundle.tensor import Tensor


class TestTensor:
    def test_gen_type(self):
        M = Manifold[2]
        V = VectorBundle[2]

        T = Tensor[V][M]
        T(torch.tensor([1.0, 2.0]))


class TestTensorOps:
    def test_add(self):
        M = Manifold[2]
        V1 = VectorBundle[3, M]

        v1 = Tensor[V1](torch.tensor([1.0, 2.0, 3.0]))
        v2 = Tensor[V1](torch.tensor([4.0, 5.0, 6.0]))

        assert_close((v1 + v2).components, torch.tensor([5.0, 7.0, 9.0]))

        # different bundle ranks

        V2 = VectorBundle[4, M]
        v3 = Tensor[V2](torch.tensor([7.0, 8.0, 9.0, 10.0]))

        with pytest.raises(ValueError):
            _ = v1 + v3

        # different base manifold

        M2 = Manifold[2]
        V3 = VectorBundle[3, M2]

        v4 = Tensor[V3](torch.tensor([4.0, 5.0, 6.0]))
        with pytest.raises(ValueError):
            _ = v1 + v4

    def test_add_compatible(self):
        M = Manifold[2]
        V = VectorBundle[3, M]
        v = Tensor[V](torch.tensor([1.0, 2.0, 3.0]))

        # compatible tensor product bundle
        TPB = TensorProductBundle[V]
        v_tpb = Tensor[TPB](torch.tensor([4.0, 5.0, 6.0]))
        assert_close((v + v_tpb).components, torch.tensor([5.0, 7.0, 9.0]))

        # incompatible tensor product bundle
        TPB2 = TensorProductBundle[V, V]
        v_tpb2 = Tensor[TPB2](torch.eye(3))

        with pytest.raises(ValueError):
            _ = v + v_tpb2

        # compatible kbundle
        KB = KBundle[1, V]
        v_kb = Tensor[KB](torch.tensor([4.0, 5.0, 6.0]))
        assert_close((v + v_kb).components, torch.tensor([5.0, 7.0, 9.0]))

        # incompatible kbundle
        KB2 = KBundle[2, V]
        v_kb2 = Tensor[KB2](torch.eye(3))

        with pytest.raises(ValueError):
            _ = v + v_kb2

        # compatible tensor bundle
        T = TangentBundle[M]
        v_t = Tensor[T](torch.tensor([1.0, 2.0]))

        TB = TensorBundle[1, 0][M]
        v_tb = Tensor[TB](torch.tensor([2.0, 3.0]))
        assert_close((v_t + v_tb).components, torch.tensor([3.0, 5.0]))

        # incompatible tensor bundle
        TB2 = TensorBundle[1, 1][M]
        v_tb2 = Tensor[TB2](torch.eye(2))
        with pytest.raises(ValueError):
            _ = v_t + v_tb2

    def test_sub(self):
        M = Manifold[2]
        V1 = VectorBundle[3, M]

        v1 = Tensor[V1](torch.tensor([1.0, 2.0, 3.0]))
        v2 = Tensor[V1](torch.tensor([4.0, 5.0, 6.0]))

        assert_close((v1 + v2).components, torch.tensor([5.0, 7.0, 9.0]))

    def test_mul(self):
        M = Manifold[2]
        V = VectorBundle[3, M]
        S = VectorBundle[0, M]

        v = Tensor[V](torch.tensor([1.0, 2.0, 3.0]))
        s = Tensor[S](torch.tensor(1.5))

        assert_close((v * s).components, torch.tensor([1.5, 3.0, 4.5]))
        assert_close((v * 1.5).components, torch.tensor([1.5, 3.0, 4.5]))

        assert_close((s * v).components, torch.tensor([1.5, 3.0, 4.5]))
        assert_close((1.5 * v).components, torch.tensor([1.5, 3.0, 4.5]))
