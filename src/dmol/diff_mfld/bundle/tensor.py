import torch

from typing import Tuple

from dmol.diff_mfld.util import classproperty, PartialSpec, DerivedPartialSpec, specs_match, specifications
from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.bundle.vector_bundle import VectorBundle, TangentBundle, CotangentBundle, TensorProductBundle


def _are_bundles_equiv(b1: tuple[type[VectorBundle], ...], b2: tuple[type[VectorBundle], ...]):
    for first, second in zip(b1, b2):
        if issubclass(first, TensorProductBundle) or issubclass(second, TensorProductBundle):
            raise TypeError()  # should be unreachable
        elif not specs_match(first, second):
            return False
    return True


def _get_most_general_bundle(b1: type[VectorBundle], b2: type[VectorBundle]):
    if issubclass(b1, b2):
        return b2
    elif issubclass(b2, b1):
        return b1
    return b1


@specifications(fields={"_bundle"})
class Tensor(metaclass=PartialSpec):
    _bundle: type[VectorBundle] | type[TensorProductBundle]
    _shape: Tuple[int, ...]

    # NOTE: restricted by torch implementation but for future rust development any dimensions of a multi-dimensional
    # array that are 0 should just be treated as a scalar (but noting that it keeping each specific index preserves
    # which of the constituent bundles are scalar bundles)

    @staticmethod
    def _compute_shape(bundle: type[VectorBundle] | type[TensorProductBundle]):
        # remove the zeros to form the realizable size

        if issubclass(bundle, TensorProductBundle):
            return tuple([prod_bundle.rank for prod_bundle in bundle.bundles if prod_bundle.rank > 0])
        elif bundle.rank == 0:
            return tuple()
        return (bundle.rank,)

    def __class_getitem__(cls, args):
        bundle: type[VectorBundle] = args

        if not issubclass(bundle, VectorBundle):
            raise TypeError()

        shape = Tensor._compute_shape(bundle)
        namespace = {"_bundle": bundle, "_shape": shape}
        if bundle.incomplete:
            namespace.update({"__class_getitem__": cls._spec_incomplete_base})

        return DerivedPartialSpec(
            f"Tensor[{bundle.__name__}]",
            (cls,),
            namespace,
        )

    @staticmethod
    def _spec_incomplete_base(dcls, args):
        underlying_manifold: type[Manifold] = args
        if underlying_manifold.incomplete:
            raise TypeError("manifold type must be fully specialized")

        upd_bundle = dcls._bundle[underlying_manifold]
        shape = Tensor._compute_shape(upd_bundle)
        return DerivedPartialSpec(
            f"Tensor[{upd_bundle.__name__}]",
            (dcls,),
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

    def __neg__(self):
        return self.__class__(-self.components)

    def __add__(self, other):
        if isinstance(other, Tensor):
            self_bundle, other_bundle = type(self).bundle, type(other).bundle
            if issubclass(self_bundle, TensorProductBundle) and issubclass(other_bundle, TensorProductBundle):
                self_bundles, other_bundles = self_bundle.bundles, other_bundle.bundles
            elif issubclass(self_bundle, TensorProductBundle):
                self_bundles, other_bundles = self_bundle.bundles, (other_bundle,)
            elif issubclass(other_bundle, TensorProductBundle):
                self_bundles, other_bundles = (self_bundle,), other_bundle.bundles
            else:
                # both are tensors instantiated off a vector bundle
                if not specs_match(self_bundle, other_bundle):
                    raise ValueError("bundles must be equivalent")
                return Tensor[self_bundle](self.components + other.components)

            if _are_bundles_equiv(self_bundles, other_bundles):
                bundle_ty = _get_most_general_bundle(self_bundle, other_bundle)  # type: ignore
                return Tensor[bundle_ty](self.components + other.components)
            raise ValueError("incompatible bundles cannot be added")
        raise NotImplemented()

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if other.bundle.rank == 0:
                return self.__class__(self.components * other.components)
            elif self.bundle.rank == 0:
                return other.__class__(self.components * other.components)
            else:
                raise ValueError()
        elif isinstance(other, float):
            return self.__class__(self.components * other)
        raise NotImplemented()

    def __rmul__(self, other):
        return self.__mul__(other)


Vec = Tensor[TangentBundle]
Cov = Tensor[CotangentBundle]


def check_tensor_type(tensor: Tensor, req_spec_tensor_type: type[Tensor]):
    tensor_type = type(tensor)
    if tensor_type != req_spec_tensor_type:
        raise ValueError(f"provided tensor type {tensor_type} does not match required {req_spec_tensor_type}")
