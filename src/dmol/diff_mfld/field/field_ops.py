from typing import override

import torch

from dmol.diff_mfld.bundle.vector_bundle import _get_compatible_bundle
from dmol.diff_mfld.connection.covar_diff import FieldCustomCovar
from dmol.diff_mfld.field import Field
from dmol.diff_mfld.field.field_types import ScalarField
from dmol.diff_mfld.field.field_types import VectorField


def _shared_field_mul(lhs: Field | float, rhs: Field | float):
    are_fields = (isinstance(lhs, Field), isinstance(rhs, Field))
    match are_fields:
        case (True, True):
            are_scalars = (
                ScalarField[lhs.bundle.base].compatible_field(lhs),  # type: ignore
                ScalarField[rhs.bundle.base].compatible_field(rhs),  # type: ignore
            )

            match are_scalars:
                case (True, True):
                    return _MulField[ScalarBundle[lhs.bundle.base]](lhs, rhs)  # type: ignore
                case (True, False):
                    return _MulField[rhs.bundle](lhs, rhs)  # type: ignore
                case (False, True):
                    return _MulField[lhs.bundle](lhs, rhs)  # type: ignore
                case _:
                    raise ValueError("at least one field must be compatible to a ScalarField")
        case (True, False):
            return _MulField[lhs.bundle](lhs, rhs)  # type: ignore
        case (False, True):
            return _MulField[rhs.bundle](lhs, rhs)  # type: ignore
        case _:
            raise NotImplemented()


def _field__add__(self, other):
    if isinstance(other, Field):
        result_bundle = _get_compatible_bundle(self.bundle, other.bundle)
        return _AddField[result_bundle](self, other)
    raise NotImplemented()


def _field__sub__(self, other):
    if isinstance(other, Field):
        result_bundle = _get_compatible_bundle(self.bundle, other.bundle)
        return _SubField[result_bundle](self, other)
    raise NotImplemented()


def _field__mul__(self, other):
    lhs, rhs = self, other  # for clarity
    return _shared_field_mul(lhs, rhs)


def _field__rmul__(self, other):
    lhs, rhs = other, self
    return _shared_field_mul(lhs, rhs)


class _AddField(FieldCustomCovar):
    def __init__(self, field: Field, other: Field):
        super().__init__()
        self._field = field
        self._other = other

    def __repr__(self) -> str:
        return f"_AddField[{self._field}, {self._other}]"

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return self._field(p).components + self._other(p).components

    @override
    def covar(self, vf: VectorField, conn) -> Field:
        raise NotImplementedError()

    @override
    def total_covar(self, conn) -> Field:
        field_covar = conn.total_covar(self._field)
        other_covar = conn.total_covar(self._other)

        return _AddField[field_covar.bundle](field_covar, other_covar)  # type: ignore


class _SubField(FieldCustomCovar):
    def __init__(self, field: Field, other: Field):
        super().__init__()
        self._field = field
        self._other = other

    def __repr__(self) -> str:
        return f"_SubField[{self._field}, {self._other}]"

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return self._field(p).components - self._other(p).components

    @override
    def covar(self, vf: VectorField, conn) -> Field:
        raise NotImplementedError()

    @override
    def total_covar(self, conn) -> Field:
        field_covar = conn.total_covar(self._field)
        other_covar = conn.total_covar(self._other)

        return _SubField(field_covar, other_covar)  # type: ignore


class _PermuteField(FieldCustomCovar):
    pass


class _ProductField(FieldCustomCovar):
    pass


class _MulField(FieldCustomCovar):
    def __init__(self, lhs: Field | float, rhs: Field | float):
        super().__init__()
        self._are_fields = (not type(lhs) is float, not type(rhs) is float)
        if self._are_fields == (True, True):
            ScalarField[lhs.bundle.base].compatible_field(lhs)  # type: ignore
            ScalarField[lhs.bundle.base].compatible_field(rhs)  # type: ignore

        self._lhs = lhs
        self._rhs = rhs

    @property
    def _check_are_fields(self) -> tuple[bool, bool]:
        return (not type(self._lhs) is float, not type(self._rhs) is float)

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        match self._check_are_fields:
            case (True, True):
                return self._lhs(p).components * self._rhs(p).components  # type: ignore
            case (True, False):
                return self._lhs(p).components * self._rhs  # type: ignore
            case (False, True):
                return self._lhs * self._rhs(p).components  # type: ignore
            case _:
                raise RuntimeError()  # not reachable

    @override
    def _eval_partials(self, p: torch.Tensor) -> torch.Tensor:
        match self._check_are_fields:
            case (True, True):
                term_1 = self._lhs.partials(p) * self._rhs(p)  # type: ignore
                term_2 = self._rhs(p) * self._rhs.partials(p)  # type: ignore
                return term_1 + term_2  # type: ignore
            case (True, False):
                return self._lhs._eval_partials(p) * self._rhs  # type: ignore
            case (False, True):
                return self._lhs * self._rhs._eval_partials(p)  # type: ignore
            case _:
                raise RuntimeError()  # not reachable

    @override
    def covar(self, vf: VectorField, conn) -> Field:
        raise NotImplementedError()

    @override
    def total_covar(self, conn) -> Field:
        match self._check_are_fields:
            case (True, True):
                term_1 = conn.total_covar(self._lhs) * self._rhs  # type: ignore
                term_2 = self._lhs * conn.total_covar(self._rhs)  # type: ignore
                return term_1 + term_2
            case (True, False):
                return conn.total_covar(self._lhs) * self._rhs  # type: ignore
            case (False, True):
                return self._lhs * conn.total_covar(self._rhs)  # type: ignore
            case _:
                raise RuntimeError()  # not reachable

    def __repr__(self) -> str:
        return f"_MulField[{self._lhs}, {self._rhs}]"
