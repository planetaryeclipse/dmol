import torch

from typing import Tuple

from dmol.diff_mfld.util import classproperty, PartialSpec, DerivedPartialSpec, specs_match, specifications
from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.bundle.vector_bundle import VectorBundle, TangentBundle, CotangentBundle, TensorProductBundle


@specifications(fields={"_bundle"})
class Tensor(metaclass=PartialSpec):
    _bundle: type[VectorBundle]
    _shape: Tuple[int, ...]

    def __class_getitem__(cls, args):
        bundle: type[VectorBundle] = args

        if issubclass(bundle, TensorProductBundle):
            shape = tuple([prod_bundle.rank for prod_bundle in bundle.bundles])
        else:
            shape = bundle.rank

        namespace = {"_bundle": bundle, "_shape": shape}
        if bundle.incomplete:
            namespace.update({"__class_getitem__": cls._spec_incomplete_base})

        return PartialSpec(
            f"Tensor[{bundle.__name__}]",
            (cls,),
            namespace,
            creating_derived=True,
        )

    @classmethod
    def _spec_incomplete_base(cls, args):
        underlying_manifold: type[Manifold] = args
        if underlying_manifold.incomplete:
            raise TypeError("manifold type must be fully specialized")

        upd_bundle = cls._bundle[underlying_manifold]
        if issubclass(upd_bundle, TensorProductBundle):
            shape = tuple([prod_bundle.rank for prod_bundle in upd_bundle.bundles])
        else:
            shape = upd_bundle.rank

        return DerivedPartialSpec(
            f"Tensor[{upd_bundle.__name__}]",
            (cls,),
            {"_bundle": upd_bundle, "_shape": shape},
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

    @classmethod
    def validate_tensor(cls, t: Tensor):
        if cls.incomplete:
            raise TypeError("tensor type must be fully specialized to validate instances")
        elif not specs_match(cls, type(t)):
            raise ValueError(f"instance with bundle {t.bundle} must match bundle {cls.bundle}")


Vec = Tensor[TangentBundle]
Cov = Tensor[CotangentBundle]


def check_tensor_type(tensor: Tensor, req_spec_tensor_type: type[Tensor]):
    tensor_type = type(tensor)
    if tensor_type != req_spec_tensor_type:
        raise ValueError(f"provided tensor type {tensor_type} does not match required {req_spec_tensor_type}")
