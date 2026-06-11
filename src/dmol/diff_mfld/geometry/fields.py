from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Tuple

import torch
from dataclasses import dataclass

from dmol.diff_mfld.geometry.metric import LeviCivitaConnection


class Manifold:
    def __init__(self):
        raise TypeError("instantiation of manifold not permitted")

    def __class_getitem__(cls, args):
        name, dim = args

        assert type(name) is str
        assert type(dim) is int and dim >= 0

        return type(
            f"Manifold_{name}{{{dim}}}",
            (cls,),
            {
                "name": name,
                "dim": dim,
            },
        )


class VectorBundle(Manifold):
    def __init__(self):
        raise TypeError("instantiation of the vector bundle not permitted")

    def __class_getitem__(cls, args):
        name, base, rank = args

        assert type(name) is str
        assert issubclass(base, Manifold)
        assert type(rank) is int and rank >= 0

        return type(
            f"VectorBundle_{name}{{{rank}}} → {base.__name__}",
            (cls,),
            {
                "name": name,
                "dim": rank,  # note that dim must be included
                "base": base,
                "rank": rank,
            },
        )


# class TangentBundle(VectorBundle):
#     def __init__(self):
#         raise TypeError("instantiations of tangent bundle not permitted")

#     def __class_getitem__(cls, args):
#         base = args
#         assert issubclass(base, Manifold)

#         tb_name = f"T{base.__name__}"
#         tb_rank = base.dim

#         return type(
#             f"TangentBundle_{tb_name}",
#             (cls,),
#             {"name": tb_name, "dim": tb_rank, "base": base, "rank": tb_rank},
#         )


# class CotangentBundle(VectorBundle):
#     def __init__(self):
#         raise TypeError("instantiations of cotangent bundle not permitted")

#     def __class_getitem__(cls, args):
#         base = args
#         assert issubclass(base, Manifold)

#         ctb_name = f"T^*{base.__name__}"
#         ctb_rank = base.dim

#         return type(
#             f"CotangentBundle_{ctb_name}",
#             (cls,),
#             {"name": ctb_name, "dim": ctb_rank, "base": base, "rank": ctb_rank},
#         )


class TensorBundle(VectorBundle):
    def __init__(self):
        raise TypeError("instantiations of tensor bundle not permitted")

    def __class_getitem__(cls, args):
        base, contra, covar = args[:3]

        assert issubclass(base, Manifold)
        assert type(contra) is int and contra >= 0
        assert type(covar) is int and covar >= 0

        # NOTE: permits custom name for the type
        tb_name = f"T^({contra},{covar})" if len(args) < 4 else args[3]
        tb_rank = base.dim ** (contra + covar)

        return type(
            tb_name,
            (cls,),
            {
                "name": tb_name,
                "dim": tb_rank,
                "base": base,
                "rank": tb_rank,  # total isomorphic vector space dimension
                "contravariant_rank": contra,  # copies of vector space
                "covariant_rank": covar,  # copies of covector space
            },
        )


class TangentBundle(VectorBundle):
    def __init__(self):
        raise TypeError("instantiations of tangent bundle not permitted")

    def __class_getitem__(cls, args):
        base = args
        assert issubclass(base, Manifold)

        tb_name = f"T{base.__name__}"
        tb_rank = base.dim

        return type(
            f"TangentBundle_{tb_name}",
            (cls,),
            {"name": tb_name, "dim": tb_rank, "base": base, "rank": tb_rank},
        )


class CotangentBundle(VectorBundle):
    def __init__(self):
        raise TypeError("instantiations of cotangent bundle not permitted")

    def __class_getitem__(cls, args):
        base = args
        assert issubclass(base, Manifold)

        ctb_name = f"T^*{base.__name__}"
        ctb_rank = base.dim

        return type(
            f"CotangentBundle_{ctb_name}",
            (cls,),
            {"name": ctb_name, "dim": ctb_rank, "base": base, "rank": ctb_rank},
        )


T_Manifold = TypeVar("T_Manifold", bound=Manifold)
T_VectorBundle = TypeVar("T_VectorBundle", bound=VectorBundle)


@dataclass
class Fiber(Generic[T_VectorBundle]):
    base_point: torch.Tensor


class Tensor(Generic[T_VectorBundle]):
    def __init__(self, base_point: torch.Tensor, components: torch.Tensor):
        self._fiber = Fiber[T_VectorBundle](base_point)
        self._components = components

    pass


class Section(ABC, Generic[T_VectorBundle]):
    @abstractmethod
    def eval(*args: *Tuple[torch.Tensor]) -> torch.Tensor:
        pass

    pass


Field = Section  # differing nomenclature when appropriate

# subclass the metric as a specific instance of a field


class Metric(
    Generic[T_Manifold],
    Tensor[TensorBundle[T_Manifold, 0, 2, f"Metric{T_Manifold.__name__}"]],
):
    pass


class MetricField(
    Generic[T_Manifold],
    Field[TensorBundle[T_Manifold, 0, 2, f"Metric{T_Manifold.__name__}"]],
):
    def eval():
        pass


# # TODO: possibly implement the following more advanced cases


# class DirectSumBundle(VectorBundle):
#     pass


# class TensorProductBundle(VectorBundle):
#     pass


mfld_u = Manifold["U", 3]
print(mfld_u)
print(mfld_u.dim)
print(mfld_u.name)

e_vb_on_u = VectorBundle["E", mfld_u, 2]
print(e_vb_on_u)

# # NOTE: currently no actual use for a nonlinear bundle at the moment but we'll subclass the vector bundle for now


# # class Bundle(Manifold):
# #     def __init__(self, name, dim, base: Manifold):
# #         super().__init__(name, dim)
# #         self._base = base

# #     def __repr__(self):
# #         return f"{super().__repr__()} → {self._base}"

# #     @property
# #     def base(self):
# #         return self._base


# class VectorBundle(ABC):
#     def __init__(self, name, dim, rank):
#         super().__init__(name, dim)
#         self._rank = rank
#         self._dual = DualBundle(self)

#     @property
#     def rank(self):
#         return self._rank

#     @abstractmethod
#     def dual(self) -> DualBundle:


# T_Bundle = TypeVar("T_Bundle", bound=Bundle)


# @dataclass
# class Fiber(Generic[T_Bundle]):
#     base_point: torch.Tensor
#     bundle: T_Bundle


# class TensorView(Generic[T_Bundle]):
#     def __init__(self, components: torch.Tensor, fiber: Fiber[T_Bundle]):
#         self._components = components
#         self._fiber = fiber

#     @property
#     def components(self):
#         return self._components

#     @property
#     def fiber(self):
#         return self._fiber

#     def __call__(self):
#         # convenient method to get the components
#         return self.components


# class DualBundle(VectorBundle):
#     """describes the functionals on a particular vector bundle"""

#     def __init__(self, vb: VectorBundle):
#         super().__init__(vb.name, vb.dim, vb.rank)
#         self._vb = vb

#     def dual(self) -> VectorBundle:
#         return self._vb

#     pass


# class TangentBundle(VectorBundle):
#     """describes the space of tangent directions on a given manifold"""

#     pass


# class Tensor:
#     """contains only the tensors evaluated at a specific point"""

#     _specialized = False

#     def __class_getitem__(cls, args):
#         contra, covar = args
#         return type(
#             f"Tensor[{contra},{covar}]",
#             (cls,),
#             {
#                 "_specialized": True,
#                 "_contravariant_rank": contra,
#                 "_covariant_rank": covar,
#             },
#         )

#     def __init__(self):
#         if not self._specialized:
#             raise TypeError("tensor must have specialized sizes")

#     def stuff(self):
#         print(f"({self._contravariant_rank},{self._covariant_rank})")
#         pass


# # class Section(Generic[T_Manifold], Generic[T_Bundle]):
# #     """contains the maps to produce a tensor at a specific point"""

# #     def __call__(self, *args, **kwds):
# #         return super().__call__(*args, **kwds)


# U = Manifold["U", 2]
# print(f"{Manifold["U", 2]}")

# # Tensor02 = Tensor[0, 2]
# # Tensor03 = Tensor[0, 3]
# # TensorField02 = Field[Tensor02]

# # print(Tensor02().stuff())
# # print(Tensor03().stuff())
# # print(Tensor().stuff())


# # class Curvature(Field[Tensor[0, 2]]):
# #     def __init__(self, levi_civita_conn: LeviCivitaConnection):
# #         # TODO: create curvature compatible with any connection

# #         pass
