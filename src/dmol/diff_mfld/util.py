import torch
from abc import abstractmethod
from typing import Dict


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)


class PartialSpec(type):
    def __new__(
        mcls,
        name,
        bases,
        namespace,
        /,
        creating_derived=False,
        top_level_type=None,
        no_spec_match=False,
        **kwds,
    ):
        # type is incomplete if any of the namespace objects are none or an object is also marked incomplete
        incomplete = True
        if creating_derived:
            # print(f"namespace: {namespace}")

            name_incomplete = len([v for k, v in namespace.items() if v is None]) > 0
            sub_partial_spec_names_incomplete = False
            for _, v in namespace.items():
                if isinstance(v, PartialSpec) and v._incomplete:
                    sub_partial_spec_names_incomplete = True

            # print(f"name_incomplete: {name_incomplete}")
            # print(
            #     f"sub_partial_spec_names_incomplete: {sub_partial_spec_names_incomplete}"
            # )

            incomplete = name_incomplete or sub_partial_spec_names_incomplete

        obj = super().__new__(mcls, name, bases, namespace, **kwds)
        obj._incomplete = incomplete
        obj._specification = namespace
        obj._top_level_type = type(obj) if top_level_type is None else top_level_type
        obj._no_spec_match = no_spec_match

        return obj

    def __call__(self, *args, **kwds):
        # print(f"self: {self}")
        # print(f"bundle: {self._bundle}")

        if self._incomplete:
            raise TypeError(f"type must be fully specified before instantiation")
        return super().__call__(*args, **kwds)

    @property
    def incomplete(self):
        # incomplete property is added by the metaclass in instantiation above
        return self._incomplete


def specs_match(ty: PartialSpec, other_ty: PartialSpec):
    if ty is not other_ty and ty._no_spec_match:
        return False
    elif ty._top_level_type is other_ty._top_level_type:
        # match all the elements
        spec, other_spec = ty._specification, other_ty._specification

        print(f"spec: {spec}")
        print(f"other_spec: {other_spec}")

        if len(spec) == len(other_spec):
            print("spec lengths equal")

            for key in spec.keys():
                field, other_field = spec[key], other_spec[key]

                if isinstance(field, PartialSpec) and isinstance(
                    other_field, PartialSpec
                ):
                    print(f"fields are spec")
                    if not specs_match(field, other_field):
                        return False
                elif not isinstance(field, PartialSpec) and not isinstance(
                    other_field, PartialSpec
                ):
                    print(f"fields are not spec")
                    if field != other_field:
                        return False
                else:
                    return False
            return True
    return False


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
