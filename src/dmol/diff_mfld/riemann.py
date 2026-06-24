import torch

from typing import Union, Callable, override

from dmol.diff_mfld.mfld import Point
from dmol.diff_mfld.bundle.tensor import Tensor, Cov, Vec, check_tensor_type
from dmol.diff_mfld.bundle.vector_bundle import TensorBundle
from dmol.diff_mfld.bundle.field import Field, LambdaField
from dmol.diff_mfld.connection.connection import TangentConnection


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


class Metric(Tensor[TensorBundle[0, 2]]):
    def __init__(self, components: torch.Tensor):
        super().__init__(components)
        self._inv = torch.linalg.inv(components)

    def sharp(self, u: Cov) -> Vec:
        Cov[self.bundle.base].validate_tensor(u)
        return Vec[self.bundle.base](self._inv @ u.components)

    def flat(self, u: Vec) -> Cov:
        Vec[self.bundle.base].validate_tensor(u)
        return Cov[self.bundle.base](self.components @ u.components)

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

    def levi_civita(self) -> LeviCivitaConnection:
        return LeviCivitaConnection[self.tensor.bundle.base](self)


class MetricLambdaField(LambdaField, MetricField):
    def __init__(self, field_fn: Callable[[torch.Tensor | tuple[torch.Tensor, ...]], torch.Tensor]):
        super().__init__(field_fn=field_fn)


class EuclideanMetricField(MetricField):
    @override
    def _eval(self, p: torch.Tensor):
        return torch.eye(self.tensor.bundle.base.dim)
