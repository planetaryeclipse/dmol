import torch

from typing import Tuple

from dmol.diff_mfld.util import classproperty, PartialSpec
from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.geometry.vector_bundle import (
    VectorBundle,
    TangentBundle,
    CotangentBundle,
)


class Tensor(metaclass=PartialSpec):
    _bundle: type[VectorBundle]
    _shape: Tuple[int, ...]

    def __class_getitem__(cls, args):
        bundle: type[VectorBundle] = args

        shape = tuple([bundle.rank for bundle in bundle.bundles])

        namespace = {"_bundle": bundle, "_shape": shape}
        if bundle.incomplete:
            namespace.update(
                {"__class_getitem__": Tensor._partial_spec_mfld_pass_to_base}
            )

        return PartialSpec(
            f"Tensor[{bundle.__name__}]",
            (cls,),
            namespace,
            creating_derived=True,
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        underlying_manifold: type[Manifold] = args
        if underlying_manifold.incomplete:
            raise TypeError("manifold type must be fully specialized")

        upd_bundle = cls._bundle[underlying_manifold]
        shape = tuple([bundle.rank for bundle in upd_bundle.bundles])

        return PartialSpec(
            f"Tensor[{upd_bundle.__name__}]",
            (cls,),
            {"_bundle": upd_bundle, "_shape": shape},
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )

    def __init__(self, components: torch.Tensor):
        if components.shape != self._shape:
            raise ValueError("components must match tensor shape")
        self._components = components

    @classproperty
    def bundle(cls):
        return cls._bundle

    @classproperty
    def shape(cls):
        return cls._shape

    @property
    def components(self):
        return self._components


Vec = Tensor[TangentBundle]
Cov = Tensor[CotangentBundle]


def check_tensor_type(tensor: Tensor, req_spec_tensor_type: type[Tensor]):
    tensor_type = type(tensor)
    if tensor_type != req_spec_tensor_type:
        raise ValueError(
            f"provided tensor type {tensor_type} does not match required {req_spec_tensor_type}"
        )
