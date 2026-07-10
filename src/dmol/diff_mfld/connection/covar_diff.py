from abc import abstractmethod
from typing import override, TYPE_CHECKING

import torch

from dmol.diff_mfld.bundle.vector_bundle import (
    CotangentBundle,
    ScalarBundle,
    TensorProductBundle,
    VectorBundle,
)

from dmol.diff_mfld.field.base import Field
from dmol.diff_mfld.field.field_types import CovectorField, ScalarField, VectorField

if TYPE_CHECKING:
    from dmol.diff_mfld.connection import Connection


def _connection_covar(self, field: Field | FieldCustomCovar, vf: VectorField) -> Field:
    if isinstance(field, FieldCustomCovar):
        return field.covar(vf, self)
    return self._covar(field, vf)


def _connection_total_covar(self, field: Field | FieldCustomCovar) -> Field:
    if isinstance(field, FieldCustomCovar):
        return field.total_covar(self)
    return self._total_covar(field)


class FieldCustomCovar(Field):
    @abstractmethod
    def covar(self, vf: VectorField, conn) -> Field:
        raise NotImplementedError()

    @abstractmethod
    def total_covar(self, conn: Connection) -> Field:
        raise NotImplementedError()


def _total_covar_bundle_ty(bundle: type[VectorBundle]) -> type[VectorBundle]:
    bundles = TensorProductBundle.product_bundles(bundle)
    if not isinstance(bundles, tuple):
        # special handling of scalar functions
        if issubclass(bundles, ScalarBundle):
            return CotangentBundle[bundle.base]
        else:
            bundles = (bundles,)  # wrap single type in a list
    bundles = list(bundles)
    bundles.append(CotangentBundle[bundle.base])

    return TensorProductBundle[bundles]


class _TotalCovarField(FieldCustomCovar):
    def __init__(self, field: Field, conn):
        super().__init__()
        self._field = field
        self._conn = conn

        # TODO: implement the total covariant derivative for all tensor fields
        if ScalarField[field.bundle.base].compatible_field(field):
            self._eval_fn = _TotalCovarField._scalar_eval
        elif CovectorField[field.bundle.base].compatible_field(field):
            self._eval_fn = _TotalCovarField._cotangent_eval
        else:
            raise ValueError(
                "automatic total covariant derivatives other than scalar or covectors not currently supported"
            )

    @staticmethod
    def _scalar_eval(p: torch.Tensor, field: Field, conn) -> torch.Tensor:
        return field._eval_partials(p)

    @staticmethod
    def _cotangent_eval(p: torch.Tensor, field: Field, conn) -> torch.Tensor:
        return field._eval_partials(p) - torch.einsum("k,kji->ij", field._eval(p), conn.coeffs(p))

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        return self._eval_fn(p, self._field, self._conn)

    @override
    def total_covar(self, conn: Connection) -> Field:
        return _TotalCovarField.create_covar(self, conn)

    @staticmethod
    def create_covar(field: Field, conn: Connection):
        return _TotalCovarField[_total_covar_bundle_ty(field.bundle)](field, conn)
