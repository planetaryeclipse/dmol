import torch

from typing import override

from dmol.diff_mfld.bundle.vector_bundle import TensorBundle
from dmol.diff_mfld.connection.covar_diff import FieldCustomCovar
from dmol.diff_mfld.field.field_types import ScalarField, VectorField
from dmol.diff_mfld.mfld import Point
from dmol.diff_mfld.field.base import Field
from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.connection.methods.methods import LogMapMethod, LogMapCovarMethod
from dmol.diff_mfld.riemann import MetricField, Metric, LeviCivitaConnection, _MetricLower


class RiemSqrDist(ScalarField, FieldCustomCovar):
    def __init__(
        self,
        q: Point | torch.Tensor,
        metric: MetricField,
        log_method: LogMapMethod | None = None,
        log_covar_method: LogMapCovarMethod | None = None,
    ):
        q = Point[self.bundle.base](q)
        MetricField[self.bundle.base].validate_field(metric)

        self._q = Point[self.bundle.base](q)  # target point
        self._metric = metric
        self._conn = metric.levi_civita()

        self._log_method = log_method if log_method is not None else LogMapMethod.DEFAULT
        self._log_covar_method = log_covar_method if log_covar_method is not None else LogMapCovarMethod.DEFAULT

    @property
    def q(self):
        return self._q

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        v = self._log_method(p, self._q, self._conn)
        metric_tensor: Metric = self._metric(p)  # type: ignore
        return torch.tensor(metric_tensor.inner(v, v))

    @override
    def _eval_partials(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()  # covar manually implemented

    @override
    def covar(self, vf: VectorField, conn: Connection) -> Field:
        raise NotImplementedError()  # for now

    @override
    def total_covar(self, conn: Connection) -> Field:
        if not self._conn == conn:
            raise ValueError()

        return _MetricLower.create_lower(
            self._metric,
            -2.0
            * RiemLog[self.bundle.base](
                self._q,
                self._metric,
                log_method=self._log_method,
                log_covar_method=self._log_covar_method,
            ),  # type: ignore
        )


class RiemLog(VectorField, FieldCustomCovar):
    def __init__(
        self,
        q: Point | torch.Tensor,
        metric: MetricField,
        log_method: LogMapMethod | None = None,
        log_covar_method: LogMapCovarMethod | None = None,
    ):
        q = Point[self.bundle.base](q)
        MetricField[self.bundle.base].validate_field(metric)

        self._q = q
        self._metric = metric
        self._conn = metric.levi_civita()

        self._log_method = log_method if log_method is not None else LogMapMethod.DEFAULT
        self._log_covar_method = log_covar_method if log_covar_method is not None else LogMapCovarMethod.DEFAULT

    @override
    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        v = self._log_method(p, self._q, self._conn)
        return v.components

    @override
    def _eval_partials(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()  # covar manually implemented

    @override
    def covar(self, vf: VectorField, conn: Connection) -> Field:
        raise NotImplementedError()  # for now

    @override
    def total_covar(self, conn: Connection) -> Field:
        if not self._conn == conn:
            raise ValueError()
        return _RiemLogCovar[self.bundle.base](
            self._q, self._metric, log_method=self._log_method, log_covar_method=self._log_covar_method
        )


class _RiemLogCovar(FieldCustomCovar[TensorBundle[1, 1]]):
    def __init__(
        self,
        q: Point | torch.Tensor,
        metric: MetricField,
        log_method: LogMapMethod | None = None,
        log_covar_method: LogMapCovarMethod | None = None,
    ):
        super().__init__()

        MetricField[self.bundle.base].validate_field(metric)
        self._q = q
        self._metric = metric
        self._conn = metric.levi_civita()

        self._log_method = log_method if log_method is not None else LogMapMethod.DEFAULT
        self._log_covar_method = log_covar_method if log_covar_method is not None else LogMapCovarMethod.DEFAULT

    def _eval(self, p: torch.Tensor) -> torch.Tensor:
        log_v = self._log_method(p, self._q, self._conn)
        log_covar = self._log_covar_method(p, self._q, log_v, self._conn)
        return log_covar.components

    def _eval_partials(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()  # no further covariant differentiation (for now)
