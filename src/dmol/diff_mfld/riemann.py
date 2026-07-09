import torch

from typing import Union, Callable, override

from dmol.diff_mfld.connection import Connection
from dmol.diff_mfld.connection.covar_diff import FieldCustomCovar
from dmol.diff_mfld.connection.tangent import TangentConnection
from dmol.diff_mfld.field.field_types import CovectorField, LambdaField, VectorField
from dmol.diff_mfld.mfld import Point
from dmol.diff_mfld.bundle.tensor import Tensor, Cov, Vec
from dmol.diff_mfld.bundle.vector_bundle import (
    TangentBundle,
    TensorBundle,
    TensorProductBundle,
    CotangentBundle,
    VectorBundle,
)
from dmol.diff_mfld.field.base import Field


class RicciCurvature(Tensor[TensorBundle[0, 2]]):
    # TODO: add support to compute scalar curvature
    pass


class RiemannCurvature(Tensor[TensorBundle[0, 4]]):
    # TODO: add support to compute ricci and scalar curvature
    pass


class LeviCivitaConnection(TangentConnection):
    def __init__(self, metric_field: MetricField):
        super().__init__()
        self._metric_field = metric_field

    @override
    def _eval(self, p):
        metric: Metric = self._metric_field(p)  # type: ignore
        metric_partials: torch.Tensor = self._metric_field.partials(p)

        conn_coeffs = 0.5 * (
            torch.einsum("lr,rjk->ljk", metric.inv, metric_partials)
            + torch.einsum("lr,rkj->ljk", metric.inv, metric_partials)
            - torch.einsum("lr,jkr->ljk", metric.inv, metric_partials)
        )
        return conn_coeffs

    def riemann(self, p: Union[Point, torch.Tensor]) -> RiemannCurvature:
        p = Point[self.bundle.base](p)
        metric = self._metric_field(p).components
        conn_coeffs = self.coeffs(p)
        conn_coeff_partials = self.partials(p)

        curvature = (
            torch.einsum("lm,mjki->ijkl", metric, conn_coeff_partials)
            - torch.einsum("lm,mikj->ijkl", metric, conn_coeff_partials)
            + torch.einsum("lm,pjk,mip->ijkl", metric, conn_coeffs, conn_coeffs)
            - torch.einsum("lm,pik,mjp->ijkl", metric, conn_coeffs, conn_coeffs)
        )
        return RiemannCurvature[self.bundle.base](curvature)

    def ricci(self, p: Union[Point, torch.Tensor]) -> RicciCurvature:
        p = Point[self.bundle.base](p)
        metric: Metric = self._metric_field(p)  # type: ignore
        ricci = torch.einsum("km,kijm->ij", metric.inv, self.riemann(p).components)
        return RicciCurvature[self.bundle.base](ricci)

    def scalar(self, p: Union[Point, torch.Tensor]) -> float:
        p = Point[self.bundle.base](p)
        metric: Metric = self._metric_field(p)  # type: ignore
        scalar = torch.einsum("ij,ij", metric.inv, self.ricci(p).components)
        return scalar.item()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LeviCivitaConnection):
            raise ValueError()
        return self._metric_field is value._metric_field


class Metric(Tensor[TensorBundle[0, 2]]):
    def __init__(self, components: torch.Tensor):
        super().__init__(components)
        self._inv = torch.linalg.inv(components)

    def sharp(self, u: Cov) -> Vec:
        Cov[self.bundle.base].validate_tensor(u)
        return Vec[self.bundle.base](self._inv @ u.components)

    def flat(self, u: Vec | Tensor[TensorBundle[1, 1]]) -> Cov | Tensor[TensorBundle[0, 2]]:
        # TODO: refactor to handle general cases but this will suffice for now
        print(self.bundle.base)
        if TangentBundle[self.bundle.base].compatible_bundle(u.bundle):
            comps = self.components @ u.components
            return Cov[self.bundle.base](comps)
        elif TensorBundle[1, 1][self.bundle.base].compatible_bundle(u.bundle):
            comps = torch.einsum("ij,jk->ik", self.components, u.components)
            return Tensor[TensorBundle[0, 2]][self.bundle.base](comps)
        else:
            raise NotImplementedError(f"metric lower not automatically compatible with bundle {u.bundle}")

    def inner(self, u: Vec, v: Vec) -> float:
        Vec[self.bundle.base].validate_tensor(u)
        Vec[self.bundle.base].validate_tensor(v)
        return (u.components @ self.components @ v.components).item()

    @property
    def inv(self):
        return self._inv


# NOTE: in a rust implementation we would rather treat the metric field as an interface automatically implemented on
# all fields defined with the metric bundle (or at least a similar interface to avoid needing metric lambda field)


class MetricField(Field[Metric]):
    def __init__(self):
        super().__init__()
        if type(self) is MetricField:
            raise TypeError("can only instantiate subclasses of metric fields")

    def flat(self, vf: VectorField) -> CovectorField:
        VectorField[vf.bundle.base].validate_field(vf)
        return _MetricLower.create_lower(self, vf)  # type: ignore

    def levi_civita(self) -> LeviCivitaConnection:
        return LeviCivitaConnection[self.tensor.bundle.base](self)


class MetricLambdaField(LambdaField, MetricField):
    def __init__(self, field_fn: Callable[[torch.Tensor | tuple[torch.Tensor, ...]], torch.Tensor]):
        super().__init__(field_fn=field_fn)

    @override
    def __call__(self, p: Point | torch.Tensor) -> Metric:
        return super().__call__(p)  # type: ignore


class EuclideanMetricField(MetricField):
    @override
    def _eval(self, p: torch.Tensor):
        return torch.eye(self.tensor.bundle.base.dim)


def _metric_lower_bundle_ty(
    bundle: type[VectorBundle], index: int | None = None
) -> tuple[int, type[TensorProductBundle]]:
    bundles = TensorProductBundle.product_bundles(bundle)
    if not isinstance(bundles, tuple):
        bundles = (bundles,)
    bundles = list(bundles)

    if index is None:
        tangent_index = None
        for i, bundle in enumerate(bundles):
            if TangentBundle[bundle.base].compatible_bundle(bundle):
                if tangent_index is None:
                    tangent_index = i
                else:
                    raise ValueError(
                        "unable to infer index to lower if tensor products of more than one tangent bundle"
                    )
        if tangent_index is None:
            raise ValueError("no tangent bundle is present to be lowered")
        bundles[tangent_index] = CotangentBundle[bundle.base]
        lower_index = tangent_index
    else:
        if not TangentBundle[bundle.base].compatible_bundle(bundles[index]):
            raise ValueError("specified index must be a tangent bundle")
        bundles[index] = CotangentBundle[bundle.base]
        lower_index = index
    return lower_index, TensorProductBundle[bundles]


class _MetricLower(FieldCustomCovar):
    def __init__(self, metric: MetricField, field: Field, index: int):
        super().__init__()
        self._metric = metric
        self._field = field
        self._index = index

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        metric = self._metric._eval(p)  # avoid computing inverse

        tensor_comps = self._field._eval(p)
        lowered_comps = torch.tensordot(metric, tensor_comps, ([1], [self._index]))  # type: ignore

        return lowered_comps

    @override
    def _eval_partials(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()  # covar manually implemented

    @override
    def covar(self, vf: VectorField, conn: Connection) -> Field:
        raise NotImplementedError()  # for now

    @override
    def total_covar(self, conn: Connection) -> Field:
        return _MetricLower.create_lower(
            self._metric,
            conn.total_covar(self._field),
            index=self._index,  # lower index added at end so this remains unaffeccted
        )

    @staticmethod
    def create_lower(metric: MetricField, field: Field, index: int | None = None):
        lower_index, lower_bundle_ty = _metric_lower_bundle_ty(field.bundle, index)
        return _MetricLower[lower_bundle_ty](metric, field, lower_index)
