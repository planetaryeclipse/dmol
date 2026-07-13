from inspect import signature
from typing import Callable, override

import torch

from dmol.diff_mfld.bundle.tensor import Scalar
from dmol.diff_mfld.bundle.vector_bundle import CotangentBundle, ScalarBundle, TangentBundle
from dmol.diff_mfld.field.base import Field
from dmol.diff_mfld.mfld import Point
from dmol.diff_mfld.util import split_coords


class ScalarField(Field[ScalarBundle]):
    @override
    def __call__(self, p: Point | torch.Tensor) -> Scalar:
        return super().__call__(p)  # type: ignore

    @staticmethod
    def max(lhs: Field | float, rhs: Field | float) -> Field | float:
        raise NotImplementedError()  # to be overriden


class FloatField(ScalarField):
    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self._value)

    def _eval_partials(self, p: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(p)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    def __repr__(self) -> str:
        return f"FloatField[{self._value}]"


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

    def __repr__(self) -> str:
        return f"LambdaField[{self._tensor}]"
