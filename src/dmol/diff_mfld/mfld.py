import torch

from typing import Union, Any

from dmol.diff_mfld.util import (
    PartialSpec,
    classproperty,
    DerivedPartialSpec,
    specs_match,
    disable_matching,
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
        self.validate_point(p)
        self._p = p.p if isinstance(p, Point) else p

    def __get__(self, instance, owner):
        return self._p

    def __eq__(self, value):
        if isinstance(value, Point):
            if specs_match(self._manifold, value._manifold):
                return torch.equal(self._p, value.p)
        return False

    @property
    def p(self):
        return self._p

    @classproperty
    def manifold(cls):
        return cls._manifold

    @classmethod
    def validate_point(cls, p: Union[Point, torch.Tensor]):
        if cls.incomplete:
            raise TypeError("point type must be fully specialized to validate instances")
        elif isinstance(p, Point):
            if not specs_match(cls, type(p)):
                raise ValueError(f"manifold of typed point {p.manifold} does not match class manifold {cls.manifold}")
        elif isinstance(p, torch.Tensor):
            if len(p.shape) != 1:
                raise ValueError("provided coords must be a 1D vector")
            elif p.shape[0] != cls.manifold.dim:

                print(p.shape)

                raise ValueError(
                    "provided coords must have the same number of components equal to the manifold dimension"
                )
        else:
            raise ValueError("instance must either be a point of the same manifold or a compatible 1D array")
