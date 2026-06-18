import torch

from typing import Union

from dmol.diff_mfld.util import (
    classproperty,
    PartialSpec,
    DerivedPartialSpec,
    specs_match,
    disable_matching,
    specifications,
)


@disable_matching
class Manifold(metaclass=PartialSpec):
    _dim: int

    def __class_getitem__(cls, args):
        dim: int = args
        if type(dim) is not int:
            raise TypeError()
        elif dim < 0:
            raise TypeError()

        return DerivedPartialSpec(
            f"Manifold[{dim}]",
            (cls,),
            {"_dim": dim},
        )

    def __init__(self, name: str):
        if type(name) is not str or name == "":
            raise ValueError()
        self._name = name

    @classproperty
    def dim(cls):
        return cls._dim

    @property
    def name(self):
        return self._name


class Point(metaclass=PartialSpec):
    _manifold: type[Manifold]

    def __class_getitem__(cls, args):
        manifold: type[Manifold] = args

        if not issubclass(manifold, Manifold):
            raise TypeError()
        elif manifold.incomplete:
            raise TypeError("manifold type must be fully specialized")

        return DerivedPartialSpec(
            f"Point[{manifold.__name__}]",
            (Point,),
            {"_manifold": manifold},
        )

    def __init__(self, p: Union[Point, torch.Tensor]):
        if isinstance(p, Point):
            if specs_match(self._manifold, p.manifold):
                print(f"specs match")
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

    def __eq__(self, value):
        if specs_match(self, value):
            return torch.equal(self._p, value.p)

    @property
    def p(self):
        return self._p

    @classproperty
    def manifold(cls):
        return cls._manifold
