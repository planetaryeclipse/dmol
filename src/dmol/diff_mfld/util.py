import torch
import inspect

from typing import Tuple, Optional, Set, TypeVar, Callable, Sequence


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)


def top_level(cls):
    cls._top_level_type = cls
    return cls


def disable_matching(cls):
    cls._enable_spec_match = False
    return cls


def _get_mro_annotations(cls: type) -> Set[str]:
    all_annotations = set()
    for base in inspect.getmro(cls):
        for key in inspect.get_annotations(base).keys():
            all_annotations.add(key)
    return all_annotations


def _get_top_level_ty(cls: type) -> type:
    top_level_ty = None
    for cls in inspect.getmro(cls):
        if issubclass(type(cls), PartialSpec):
            top_level_ty = cls
    if top_level_ty is None:
        raise TypeError("at least one class in the mro must have metaclass PartialSpec")
    return top_level_ty


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
        cls._incomplete = True  # type: ignore
        cls._allow_incomplete = getattr(cls, "_allow_incomplete", False)  # type: ignore

        cls._min_complete = False  # type: ignore
        cls._top_level_type = getattr(cls, "_top_level_type", _get_top_level_ty(cls))  # type: ignore
        cls._enable_spec_match = True  # type: ignore

        # default is that specs are all the annotations in mro
        all_annotations = _get_mro_annotations(cls)
        cls._specs = all_annotations  # type: ignore

        # ensures that all attributes at least have a value
        members = {key: value for key, value in inspect.getmembers(cls)}
        for spec in cls._specs:  # type: ignore
            if spec not in members:
                setattr(cls, spec, None)

        return cls

    def __call__(self, *args, **kwds):
        if self._incomplete and not self._allow_incomplete:  # type: ignore
            raise TypeError(f"type must be fully specified before instantiation")
        return super().__call__(*args, **kwds)

    @property
    def incomplete(self):
        return self._incomplete  # type: ignore

    @property
    def top_level(self):
        return self._top_level_type  # type: ignore


class DerivedPartialSpec(PartialSpec):
    def __new__(
        mcls,
        name,
        bases,
        namespace,
        /,
        **kwds,
    ):
        # pulls any previously set specs from the base classes
        upd_full_specs = {}
        for base in bases:
            if getattr(base, "_specs", None) is None:
                continue

            members = {key: value for key, value in inspect.getmembers(base)}
            for spec in base._specs:
                if spec in members:
                    if spec in upd_full_specs:
                        # override the specs if defined later in the mro
                        if members[spec] is not None:
                            upd_full_specs[spec] = members[spec]
                    else:
                        upd_full_specs[spec] = members[spec]

                    pass
                elif spec not in upd_full_specs:
                    upd_full_specs[spec] = None  # default if not in bases

        # applies any new specs from the namespace (if applicable)
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
        for key, value in upd_full_specs.items():
            setattr(obj, key, value)

        obj._incomplete = specs_incomplete  # type: ignore
        obj._top_level_type = base._top_level_type  # type: ignore
        obj._enable_spec_match = base._enable_spec_match  # type: ignore

        if not specs_incomplete:
            setattr(obj, "__class_getitem__", DerivedPartialSpec._exhausted_specs)

        return obj

    def _exhausted_specs(*args):
        raise TypeError("no further type specialization")


T = TypeVar("T", bound=PartialSpec)


def specifications(
    cls: T | None = None,
    *,
    fields: Optional[Set[str]] = None,
) -> Callable[[T], T] | T:
    def decorate(cls: T):
        all_annotations = _get_mro_annotations(cls)
        if fields is not None:
            if not fields.issubset(all_annotations):
                raise TypeError("full specifications must be found as annotations in class mro")
            cls._specs = fields  # type: ignore
        return cls

    return decorate if cls is None else decorate(cls)


def _check_fields(field, other_field) -> bool:
    if isinstance(field, PartialSpec) and isinstance(other_field, PartialSpec):
        if not specs_match(field, other_field):
            return False
    elif isinstance(field, Sequence) and isinstance(other_field, Sequence):
        for element, other_element in zip(field, other_field):
            if not _check_fields(element, other_element):
                return False
    elif field != other_field:
        return False
    return True


def specs_match(ty: PartialSpec, other_ty: PartialSpec):
    if not ty._enable_spec_match or not other_ty._enable_spec_match:  # type: ignore
        # if spec match is disabled then directly checks that the classes are the same
        return ty is other_ty
    elif ty._top_level_type is not other_ty._top_level_type:  # type: ignore
        # if the top level type is not the same then the specs are defined to be disjoint
        return False

    # recursively checks equivalence of the internal specifications of the type
    if ty._specs == other_ty._specs:  # type: ignore
        for spec in ty._specs:  # type: ignore
            field, other_field = getattr(ty, spec), getattr(other_ty, spec)
            if not _check_fields(field, other_field):
                return False
    else:
        return False

    return True


def split_coords(p: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    assert len(p.shape) == 1
    return tuple(p[i] for i in range(len(p)))


def combine_coords(*coords: torch.Tensor) -> torch.Tensor:
    for coord in coords:
        assert len(coord.shape) == 0

    p = torch.zeros((len(coords),))
    for i, coord in enumerate(coords):
        p[i] = coord

    return p
