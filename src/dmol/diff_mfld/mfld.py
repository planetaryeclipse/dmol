# import torch
# from dataclasses import dataclass
# from typing import Optional

# from dmol.diff_mfld.geometry.riemannian import MetricField
# from dmol.diff_mfld.connection.connection import Connection

# from dmol.diff_mfld.connection.geodesic_funcs import (
#     ExpMethod,
#     LogMethod,
# )


# @dataclass
# class Mfld:
#     metric: Optional[MetricField]
#     conn: Connection


# @dataclass
# class ComputeMfld:
#     mfld: Mfld

#     exp_method: ExpMethod = ExpMethod.APPROX_O2
#     log_method: LogMethod = LogMethod.APPROX_O2
#     dist_method: LogMethod = LogMethod.APPROX_O2

#     def exp(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         return self.exp_method(p, v, self.mfld.conn)

#     def log(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
#         return self.log_method(p, q, self.mfld.conn)

#     def dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
#         v = self.log_method(p, q, self.mfld.conn)
#         metric = self.mfld.metric(p)
#         return metric(v, v)


import torch

from typing import Union

from dmol.diff_mfld.util import classproperty, PartialSpec, specs_match


class Manifold(metaclass=PartialSpec):
    _dim: int = None

    def __class_getitem__(cls, args):
        dim: int
        dim = args

        return PartialSpec(
            f"Manifold[{dim}]", (cls,), {"_dim": dim}, creating_derived=True
        )

    def __init__(self, name: str):
        self._name = name

    @classproperty
    def dim(cls):
        return cls._dim

    @property
    def name(self):
        return self._name


class Point(metaclass=PartialSpec):
    _manifold: type[Manifold]

    def __class_getitem__(cls, manifold: type[Manifold]):
        if manifold.incomplete:
            raise TypeError("manifold type must be fully specialized")

        return PartialSpec(
            f"Point[{manifold.__name__}]",
            (Point,),
            {"_manifold": manifold},
            creating_derived=True,
        )

    def __init__(self, p: Union[Point, torch.Tensor]):
        if isinstance(p, Point):
            if specs_match(self._manifold, p.manifold):
                self._p = p.p
            else:
                raise ValueError(
                    "creating a point from another point must have same manifold"
                )
        else:
            if not (len(p.shape) == 1 and p.shape[0] == self._manifold.dim):
                raise ValueError(
                    "provided coords must be a 1D vector with components equal to manifold dimension"
                )
            self._p = p

    def __get__(self, instance, owner):
        return self._p

    @property
    def p(self):
        return self._p

    @classproperty
    def manifold(cls):
        return cls._manifold
