from abc import abstractmethod

import torch

from dmol.diff_mfld.bundle.tensor import Tensor, Vec
from dmol.diff_mfld.bundle.vector_bundle import TangentBundle
from dmol.diff_mfld.connection.base import Connection
from dmol.diff_mfld.connection.covar_diff import _TotalCovarField
from dmol.diff_mfld.connection.methods.geod_ivp_bvp import bvp_log_map, ivp_exp_map
from dmol.diff_mfld.connection.methods.parl_transp import ivp_parl_transp_vec
from dmol.diff_mfld.curve import Curve
from dmol.diff_mfld.field.base import Field
from dmol.diff_mfld.field.field_types import VectorField
from dmol.diff_mfld.mfld import Point


class TangentConnection(Connection[TangentBundle]):
    def _covar(self, field: Field, vf: VectorField) -> Field:
        # TODO: add operations on tensors
        raise NotImplementedError()

    def _total_covar(self, field: Field) -> Field:
        return _TotalCovarField.create_covar(field, self)

    @abstractmethod
    def exp(self, p: Point | torch.Tensor, v: Vec) -> tuple[Point, Curve]:
        Point[self.bundle.base].validate_point(p)
        Tensor[self.bundle].validate_tensor(v)
        return ivp_exp_map(p, v, self)

    @abstractmethod
    def log(
        self,
        p: Point | torch.Tensor,
        q: Point | torch.Tensor,
    ) -> tuple[Vec, Curve]:
        Point[self.bundle.base].validate_point(p)
        Point[self.bundle.base].validate_point(q)
        return bvp_log_map(p, q, self)

    @abstractmethod
    def pt_vec(self, u: Vec, curve: Curve) -> Vec:
        Tensor[self.bundle].validate_tensor(u)
        return ivp_parl_transp_vec(u, curve, self)
