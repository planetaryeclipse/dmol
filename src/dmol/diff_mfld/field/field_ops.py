from typing import Sequence, override

import torch

from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle, TensorProductBundle, VectorBundle, _get_compatible_bundle
from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.connection.covar_diff import FieldCustomCovar
from dmol.diff_mfld.field import Field
from dmol.diff_mfld.field.field_types import FloatField, ScalarField
from dmol.diff_mfld.field.field_types import VectorField


def _wrap_numerics(lhs: Field | float | int, rhs: Field | float | int):
    lhs_field: Field
    rhs_field: Field

    if type(lhs) is float or type(lhs) is int:
        rhs_field = rhs  # type: ignore
        lhs_field = FloatField[rhs_field.bundle.base](float(lhs))
    elif type(rhs) is float or type(rhs) is int:
        lhs_field = lhs  # type: ignore
        rhs_field = FloatField[lhs_field.bundle.base](float(rhs))
    else:
        lhs_field, rhs_field = lhs, rhs  # type: ignore
    return lhs_field, rhs_field


def _shared_field_add(lhs: Field | float | int, rhs: Field | float | int):
    lhs_field, rhs_field = _wrap_numerics(lhs, rhs)
    if lhs_field.compatible_field(rhs_field):
        result_bundle = _get_compatible_bundle(lhs_field.bundle, rhs_field.bundle)
        return _AddField[result_bundle](lhs_field, rhs_field)
    return NotImplemented


def _shared_field_sub(lhs: Field | float | int, rhs: Field | float | int):
    lhs_field, rhs_field = _wrap_numerics(lhs, rhs)
    if lhs_field.compatible_field(rhs_field):
        result_bundle = _get_compatible_bundle(lhs_field.bundle, rhs_field.bundle)
        return _SubField[result_bundle](lhs_field, rhs_field)
    return NotImplemented


def _shared_field_mul(lhs: Field | float | int, rhs: Field | float | int):
    lhs_field, rhs_field = _wrap_numerics(lhs, rhs)

    lhs_scalar = ScalarField[lhs_field.bundle.base].compatible_field(lhs_field)
    rhs_scalar = ScalarField[rhs_field.bundle.base].compatible_field(rhs_field)
    bundle_choice: type[VectorBundle]
    if (lhs_scalar and rhs_scalar) or rhs_scalar:
        bundle_choice = lhs_field.bundle
    elif lhs_scalar:
        bundle_choice = rhs_field.bundle
    else:
        raise RuntimeError()  # not reachable

    return _MulField[bundle_choice](lhs_field, rhs_field)


def _shared_field_div(lhs: Field | float | int, rhs: Field | float | int):
    lhs_field, rhs_field = _wrap_numerics(lhs, rhs)
    return _DivField[lhs_field.bundle](lhs_field, rhs_field)


def _field__add__(self, other):
    lhs, rhs = self, other
    return _shared_field_add(lhs, rhs)


def _field__radd__(self, other):
    lhs, rhs = other, self
    return _shared_field_add(lhs, rhs)


def _field__sub__(self, other):
    lhs, rhs = self, other
    return _shared_field_sub(lhs, rhs)


def _field__rsub__(self, other):
    lhs, rhs = other, self
    return _shared_field_sub(lhs, rhs)


def _field__mul__(self, other):
    lhs, rhs = self, other  # for clarity
    return _shared_field_mul(lhs, rhs)


def _field__rmul__(self, other):
    lhs, rhs = other, self
    return _shared_field_mul(lhs, rhs)


def _field__truediv__(self, other):
    lhs, rhs = self, other
    return _shared_field_div(lhs, rhs)


def _field__rtruediv__(self, other):
    lhs, rhs = other, self
    return _shared_field_div(lhs, rhs)


def _field__pow__(self, other):
    return _PowerField.create_power(self, other)


# fields to handle the various operators


class _AddField(FieldCustomCovar):
    def __init__(self, field: Field, other: Field):
        super().__init__()
        self._field = field
        self._other = other

    def __repr__(self) -> str:
        return f"_AddField[{self._field}, {self._other}]"

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return self._field.comps(p) + self._other.comps(p)

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
        return self._field.comps(p) - self._other.comps(p)

    @override
    def covar(self, vf: VectorField, conn) -> Field:
        raise NotImplementedError()

    @override
    def total_covar(self, conn) -> Field:
        field_covar = conn.total_covar(self._field)
        other_covar = conn.total_covar(self._other)

        return _SubField[field_covar.bundle](field_covar, other_covar)  # type: ignore


class _PermuteField(FieldCustomCovar):
    def __init__(self, field: Field, permute: Sequence[int]):
        super().__init__()
        self._field = field
        self._permute = permute

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        tens = self._field.comps(p)
        permute_tens = torch.permute(tens, self._permute)  # type: ignore
        return permute_tens

    @staticmethod
    def create_permute(field: Field, permute: Sequence[int]):
        bundles = TensorProductBundle.product_bundles(field.bundle)
        if not isinstance(bundles, Sequence):
            return field
        else:
            orig_bundles = list(bundles)
            upd_bundles = [orig_bundles[orig_idx] for orig_idx in permute]
            return _PermuteField[TensorProductBundle[upd_bundles]](field, permute)

    def __repr__(self) -> str:
        return f"_PermuteField[{self._field}, {self._permute}]"


class _ProductField(FieldCustomCovar):
    def __init__(self, fields: Sequence[Field]):
        super().__init__()
        self._fields = fields

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        result = self._fields[0].comps(p)
        for next_field in self._fields[1:]:
            result = torch.tensordot(result, next_field.comps(p), dims=0)
        return result

    @staticmethod
    def create_product(fields: Sequence[Field]):
        # obtain the resulting bundle type
        result_bundles = []
        for field in fields:
            field_bundles = TensorProductBundle.product_bundles(field.bundle)
            if isinstance(field_bundles, Sequence):
                result_bundles.extend(field_bundles)
            else:
                result_bundles.append(field_bundles)

        return _ProductField[TensorProductBundle[result_bundles]](fields)

    def __repr__(self) -> str:
        return f"_ProductField[{str.join(", ", [field.__repr__() for field in self._fields])}]"


class _MulField(FieldCustomCovar):
    def __init__(self, lhs: Field, rhs: Field):
        super().__init__()
        self._lhs = lhs
        self._rhs = rhs

        lhs_scalar, rhs_scalar = self._are_scalars
        if not lhs_scalar and not rhs_scalar:
            raise ValueError(
                "unable to perform scalar multiplication between two non-scalar fields (use tensor product)"
            )

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return self._lhs(p).components * self._rhs(p).components

    @property
    def _are_scalars(self):
        return (
            ScalarField[self._lhs.bundle.base].compatible_field(self._lhs),
            ScalarField[self._rhs.bundle.base].compatible_field(self._rhs),
        )

    @override
    def _eval_partials(self, p: torch.Tensor) -> torch.Tensor:
        lhs_scalar, rhs_scalar = self._are_scalars
        if lhs_scalar and rhs_scalar:
            return self._lhs.partials(p) * self._rhs.comps(p) + self._lhs.comps(p) * self._rhs.partials(p)
        elif lhs_scalar:
            # dimension appearing due to differentation must appear at end so cannot follow exact leibniz rule ordering
            return torch.outer(self._rhs.comps(p), self._lhs.partials(p)) + self._lhs.comps(p) * self._rhs.partials(p)
        elif rhs_scalar:
            return self._lhs.partials(p) * self._rhs.comps(p) + torch.outer(self._lhs.comps(p), self._rhs.partials(p))
        else:
            raise RuntimeError()  # not reachable

    @override
    def covar(self, vf: VectorField, conn) -> Field:
        raise NotImplementedError()

    @override
    def total_covar(self, conn) -> Field:
        lhs_scalar, rhs_scalar = self._are_scalars
        if lhs_scalar and rhs_scalar:
            return conn.total_covar(self._lhs) * self._rhs + self._lhs * conn.total_covar(self._rhs)
        elif lhs_scalar:
            # must shift covariant index of scalar field differential to end of tensor product
            prod_tensor_rank = 1 + TensorProductBundle.product_tensor_rank(self._rhs.bundle)
            term_1 = _PermuteField.create_permute(
                _ProductField.create_product((conn.total_covar(self._lhs), self._rhs)),
                (*range(1, prod_tensor_rank), 0),
            )
            term_2 = self._lhs * conn.total_covar(self._rhs)
            return term_1 + term_2
        elif rhs_scalar:
            # covariant index of scalar field differential already at correct location
            term_1 = conn.total_covar(self._lhs) * self._rhs
            term_2 = _ProductField.create_product((self._lhs, conn.total_covar(self._rhs)))
            return term_1 + term_2
        else:
            raise RuntimeError()  # not reachable

    def __repr__(self) -> str:
        return f"_MulField[{self._lhs}, {self._rhs}]"


class _DivField(FieldCustomCovar):
    def __init__(self, num: Field, den: Field):
        super().__init__()
        self._num = num
        self._den = den

        if not ScalarField[num.bundle.base].compatible_field(den):
            raise ValueError("can only divide by a scalar field")

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return self._num.comps(p) / self._den.comps(p)

    @override
    def total_covar(self, conn: Connection) -> Field:
        term_1 = conn.total_covar(self._num) / self._den
        term_2 = self._num / self._den**2 * conn.total_covar(self._den)

        return term_1 + term_2

    def __repr__(self) -> str:
        return f"_DivField[{self._num}, {self._den}]"


class _MaxField(FieldCustomCovar[ScalarBundle]):
    def __init__(self, lhs: Field, rhs: Field):
        super().__init__()
        self._lhs = lhs
        self._rhs = rhs

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        lhs = self._lhs.comps(p)
        rhs = self._rhs.comps(p)
        return lhs if lhs >= rhs else rhs

    def __repr__(self) -> str:
        return f"_MaxField[{self._lhs}, {self._rhs}]"

    @override
    def total_covar(self, conn: Connection) -> Field:
        return _MaxCovarField.create_max_covar(
            self._lhs,
            self._rhs,
            conn.total_covar(self._lhs),
            conn.total_covar(self._rhs),
        )

    @staticmethod
    def create_max(lhs: Field | float, rhs: Field | float):
        lhs_field: Field
        rhs_field: Field

        if type(lhs) is float and type(rhs) is float:
            return max(lhs, rhs)
        elif type(lhs) is float:
            rhs_field = rhs  # type: ignore
            lhs_field = FloatField[rhs_field.bundle.base](lhs)
        elif type(rhs) is float:
            lhs_field = lhs  # type: ignore
            rhs_field = FloatField[lhs_field.bundle.base](rhs)
        else:
            lhs_field = lhs  # type: ignore
            rhs_field = rhs  # type: ignore

        lhs_scalar = ScalarField[lhs_field.bundle.base].compatible_field(lhs_field)
        rhs_scalar = ScalarField[rhs_field.bundle.base].compatible_field(rhs_field)
        if not lhs_scalar or not rhs_scalar:
            raise ValueError("both arguments must be scalar fields")

        return _MaxField[lhs_field.bundle.base](lhs_field, rhs_field)


class _MaxCovarField(FieldCustomCovar):
    def __init__(self, lhs: Field, rhs: Field, lhs_covar: Field, rhs_covar: Field):
        super().__init__()
        self._lhs = lhs
        self._rhs = rhs
        self._lhs_covar = lhs_covar
        self._rhs_covar = rhs_covar

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        lhs = self._lhs.comps(p)
        rhs = self._rhs.comps(p)
        return self._lhs_covar.comps(p) if lhs >= rhs else self._rhs_covar.comps(p)

    def __repr__(self) -> str:
        return f"_MaxCovarField[{self._lhs}, {self._rhs}, {self._lhs_covar}, {self._rhs_covar}]"

    @staticmethod
    def create_max_covar(lhs: Field, rhs: Field, lhs_covar: Field, rhs_covar: Field):
        return _MaxCovarField[lhs_covar.bundle](
            lhs,
            rhs,
            lhs_covar,
            rhs_covar,
        )

    @override
    def total_covar(self, conn: Connection) -> Field:
        return _MaxCovarField.create_max_covar(
            self._lhs_covar,
            self._rhs_covar,
            conn.total_covar(self._lhs_covar),
            conn.total_covar(self._rhs_covar),
        )


class _PowerField(FieldCustomCovar[ScalarBundle]):
    def __init__(self, field: Field, power: int):
        super().__init__()

        self._field = field
        self._power = power

    def __repr__(self) -> str:
        return f"_PowerField[{self._field}, {self._power}]"

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return self._field.comps(p) ** self._power

    @override
    def total_covar(self, conn: Connection) -> Field:
        return (
            float(self._power)
            * _PowerField[self.bundle.base](self._field, self._power - 1)
            * conn.total_covar(self._field)
        )

    @staticmethod
    def create_power(field: Field, power: int):
        if not ScalarField[field.bundle.base].compatible_field(field):
            raise ValueError("power can only be applied to scalar fields")
        return _PowerField[field.bundle.base](field, power)
