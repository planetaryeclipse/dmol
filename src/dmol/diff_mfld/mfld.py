


import torch

from typing import Union

from dmol.diff_mfld.util import classproperty, PartialSpec, specs_match


class Manifold(metaclass=PartialSpec):
    _dim: int = None

    def __class_getitem__(cls, args):
        dim: int = args
        if type(dim) is not int:
            raise TypeError()
        elif dim < 0:
            raise TypeError()

        return PartialSpec(
            f"Manifold[{dim}]",
            (cls,),
            {"_dim": dim, "__class_getitem__": Manifold._exhausted},
            creating_derived=True,
            no_spec_match=True,
        )

    def _exhausted(cls, args):
        raise TypeError("no further type specialization")

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

        return PartialSpec(
            f"Point[{manifold.__name__}]",
            (Point,),
            {"_manifold": manifold},
            creating_derived=True,
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
