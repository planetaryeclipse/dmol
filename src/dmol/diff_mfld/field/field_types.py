from inspect import signature
from typing import Callable, override

import torch

from dmol.diff_mfld.bundle.tensor import Scalar, Tensor
from dmol.diff_mfld.bundle.vector_bundle import CotangentBundle, ScalarBundle, TangentBundle
from dmol.diff_mfld.field.base import Field
from dmol.diff_mfld.mfld import Point
from dmol.diff_mfld.util import split_coords


class ScalarField(Field[ScalarBundle]):
    @override
    def __call__(self, p: Point | Tensor) -> Scalar:
        return super().__call__(p)  # type: ignore


class VectorField(Field[TangentBundle]):
    pass


class CovectorField(Field[CotangentBundle]):
    pass


class LambdaField(Field):
    def __init__(self, field_fn: Callable[[torch.Tensor | tuple[torch.Tensor, ...]], torch.Tensor]):
        super().__init__()

        n = self.tensor.bundle.base.dim
        num_args = len(signature(field_fn).parameters)

        sample_p = torch.zeros((n,))

        if num_args > 1:
            if num_args != n:
                raise ValueError(f"provided function accepts {num_args} but manifold has {n} dimensions")
            self._has_coord_fn = True
            sample_components = field_fn(*split_coords(sample_p))
        else:
            self._has_coord_fn = False
            sample_components = field_fn(sample_p)

        if sample_components.shape != self.tensor.shape:
            raise ValueError()

        self._field_fn = field_fn

    @override
    def _eval(self, p: torch.Tensor):
        if self._has_coord_fn:
            components = self._field_fn(*split_coords(p))
        else:
            components = self._field_fn(p)
        return components
