import torch

from typing import Union, Self

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.geometry.tensor import Tensor, Cov, Vec, check_tensor_type
from dmol.diff_mfld.geometry.vector_bundle import TensorBundle, TangentBundle
from dmol.diff_mfld.geometry.field import Field
from dmol.diff_mfld.connection.connection import Connection


class RicciCurvature(Tensor[TensorBundle[0, 2]]):
    # TODO: add support to compute scalar curvature
    pass


class RiemannCurvature(Tensor[TensorBundle[0, 4]]):
    # TODO: add support to compute ricci and scalar curvature
    pass


class LeviCivitaConnection(Connection[TangentBundle]):
    def __init__(self, metric_field: MetricField):
        super().__init__()
        self._metric_field = metric_field

    def eval(self, p):
        print(self._metric_field(p))

        metric_inv: torch.Tensor = self._metric_field(p)._inv
        metric_partials: torch.Tensor = self._metric_field.partials(p)

        conn_coeffs = 0.5 * (
            torch.einsum("lr,rjk->ljk", metric_inv, metric_partials)
            + torch.einsum("lr,rkj->ljk", metric_inv, metric_partials)
            - torch.einsum("lr,jkr->ljk", metric_inv, metric_partials)
        )
        return conn_coeffs

    def riemann(
        self, p: Union[Point[Self._bundle.base], torch.Tensor]
    ) -> RiemannCurvature[Self._bundle.base]:
        metric = self._metric_field(p).components
        conn_coeffs = self.coeffs(p)
        conn_coeff_partials = self.partials(p)

        curvature = (
            torch.einsum("lm,mjki->ijkl", metric, conn_coeff_partials)
            - torch.einsum("lm,mikj->ijkl", metric, conn_coeff_partials)
            + torch.einsum("lm,pjk,mip->ijkl", metric, conn_coeffs, conn_coeffs)
            - torch.einsum("lm,pik,mjp->ijkl", metric, conn_coeffs, conn_coeffs)
        )
        return RiemannCurvature[self._bundle.base](curvature)

    def ricci(
        self, p: Union[Point[Self._bundle.base], torch.Tensor]
    ) -> RicciCurvature[Self._bundle.base]:
        riemann: torch.Tensor = self.riemann(p).components
        metric_inv: torch.Tensor = self._metric_field(p)._inv

        ricci = torch.einsum("km,kijm->ij", metric_inv, riemann)
        return RicciCurvature[self._bundle.base](ricci)

    def scalar(self, p: Union[Point[Self._bundle.base], torch.Tensor]) -> float:
        metric_inv: torch.Tensor = self._metric_field(p)._inv
        ricci: torch.Tensor = self.ricci(p).components

        scalar = torch.einsum("ij,ij", metric_inv, ricci)
        return scalar


class Metric(Tensor[TensorBundle[0, 2]]):
    def __init__(self, components: torch.Tensor):
        super().__init__(components)
        self._inv = torch.linalg.inv(components)

    def sharp(self, u: Cov[Self._bundle.base]) -> Vec[Self._bundle.base]:
        check_tensor_type(u, Cov[self.bundle.base])
        return Vec[self._bundle.base](self._inv @ u.components)

    def flat(self, u: Vec[Self._bundle.base]) -> Cov[Self._bundle.base]:
        check_tensor_type(u, Vec[self.bundle.base])
        return Cov[self._bundle.base](self.components @ u.components)

    def inner(self, u: Vec[Self._bundle.base], v: Vec[Self._bundle.base]) -> float:
        check_tensor_type(u, Vec[self.bundle.base])
        check_tensor_type(v, Vec[self.bundle.base])
        return u.components @ self.components @ v.components


class MetricField(Field[Metric]):
    def __init__(self):
        if type(self) is MetricField:
            raise TypeError("can only instantiate subclasses of metric fields")

    def levi_civita(self) -> LeviCivitaConnection[Self._bundle.base]:
        # print(f"metric field bundle: {self._bundle}")
        # print(f"metric field bundle bundle: {self._bundle._bundle}")

        return LeviCivitaConnection[self.tensor.bundle.base](self)


class EuclideanMetricField(MetricField):
    def eval(self, p):
        print("yeet")
        print(self._tensor)

        return torch.eye(self._tensor.bundle.base.dim)


# M2 = Manifold[2]
# Euclid = EuclideanMetricField[M2]

# print(Euclid)

# print(Euclid()(torch.tensor([1.0, 2.0]))._inv)
