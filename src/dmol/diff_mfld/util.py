import torch

import inspect

from abc import abstractmethod
from typing import Dict, Tuple, Optional, Set


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)


def disable_matching(cls):
    cls._enable_spec_match = False
    return cls


def _get_mro_annotations(cls: type) -> Set[str]:
    all_annotations = set()
    for base in inspect.getmro(cls):
        for key in inspect.get_annotations(base).keys():
            all_annotations.add(key)
    return all_annotations


def specifications(
    cls=None,
    *,
    fields: Optional[Set[str]] = None,
):
    def decorate(cls: type):
        all_annotations = _get_mro_annotations(cls)

        if fields is not None:
            if not fields.issubset(all_annotations):
                raise TypeError(
                    "full specifications must be found as annotations in class mro"
                )
            cls._specs = fields

    return decorate if cls is None else decorate(cls)


class PartialSpec(type):
    def __new__(
        mcls,
        name,
        bases,
        namespace,
        /,
        **kwds,
    ):
        cls = super().__new__(mcls, name, bases, namespace, **kwds)
        cls._incomplete = True
        cls._min_complete = False
        cls._top_level_type = cls
        cls._enable_spec_match = True

        # default is that specs are all the annotations in mro
        all_annotations = _get_mro_annotations(cls)
        cls._specs = all_annotations

        # ensures that all attributes at least have a value
        members = {key: value for key, value in inspect.getmembers(cls)}
        for spec in cls._specs:
            if spec not in members:
                setattr(cls, spec, None)

        return cls

    def __call__(self, *args, **kwds):
        if self._incomplete:
            raise TypeError(f"type must be fully specified before instantiation")
        return super().__call__(*args, **kwds)

    @property
    def incomplete(self):
        return self._incomplete

    @property
    def top_level(self):
        return self._top_level_type


class DerivedPartialSpec(PartialSpec):
    def __new__(
        mcls,
        name,
        bases,
        namespace,
        /,
        **kwds,
    ):
        (base,) = bases

        members = {key: value for key, value in inspect.getmembers(base)}
        upd_full_specs = {
            spec: (members[spec] if spec in members.keys() else None)
            for spec in base._specs
        }
        for key, value in namespace.items():
            if key in upd_full_specs:
                upd_full_specs[key] = value

        # checks whether the updated annotations are complete

        specs_incomplete = False
        for _, value in upd_full_specs.items():
            if value is None or (isinstance(value, PartialSpec) and value.incomplete):
                specs_incomplete = True
                break

        # creates the derived class

        obj = super().__new__(mcls, name, bases, namespace, **kwds)

        obj._incomplete = specs_incomplete
        obj._top_level_type = base._top_level_type
        obj._enable_spec_match = base._enable_spec_match

        if not specs_incomplete:
            obj.__class_getitem__ = DerivedPartialSpec._exhausted_specs

        return obj

    def _exhausted_specs(*args):
        raise TypeError("no further type specialization")


def specs_match(ty: PartialSpec, other_ty: PartialSpec):
    if not ty._enable_spec_match or not other_ty._enable_spec_match:
        # if spec match is disabled then directly checks that the classes are the same
        return ty is other_ty
    elif ty._top_level_type is not other_ty._top_level_type:
        # if the top level type is not the same then the specs are defined to be disjoint
        return False

    # recursively checks equivalence of the internal specifications of the type
    if ty._specs == other_ty._specs:
        for spec in ty._specs:
            field, other_field = getattr(ty, spec), getattr(other_ty, spec)

            if isinstance(field, PartialSpec) and isinstance(other_field, PartialSpec):
                if not specs_match(field, other_field):
                    return False
            elif field != other_field:
                return False
    else:
        return False

    return True


def split_coords(p: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    assert len(p.shape) == 1
    return (p[i] for i in range(len(p)))


def combine_coords(*coords: torch.Tensor) -> torch.Tensor:
    for coord in coords:
        assert len(coord.shape) == 0

    p = torch.zeros((len(coords),))
    for i, coord in enumerate(coords):
        p[i] = coord

    return p
