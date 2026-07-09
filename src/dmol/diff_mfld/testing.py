import torch

from torch.testing import assert_close
from dmol.diff_mfld.bundle.tensor import Tensor, Scalar


def _convert_ty(tensor: Tensor | float | torch.Tensor, other: Tensor | float | torch.Tensor) -> Tensor | torch.Tensor:
    if isinstance(tensor, float):
        tensor = torch.tensor(tensor)
    if isinstance(tensor, torch.Tensor):
        if len(tensor.shape) != 0:
            raise ValueError("only scalar torch tensors accepted")

        if isinstance(other, Tensor):
            tensor = Scalar[other.bundle.base](tensor)
    return tensor  # type: ignore


def assert_tensors_equiv(
    tensor: Tensor | float | torch.Tensor,
    other: Tensor | float | torch.Tensor,
    rtol: float | None = None,
    atol: float | None = None,
):
    tensor = _convert_ty(tensor, other)
    other = _convert_ty(other, tensor)

    if isinstance(tensor, torch.Tensor) and isinstance(other, torch.Tensor):
        return assert_close(tensor, other)
    elif isinstance(tensor, Tensor) and isinstance(other, Tensor):
        type(tensor).validate_tensor(other)
        assert_close(tensor.components, other.components, rtol=rtol, atol=atol)
    else:
        raise NotImplementedError()
