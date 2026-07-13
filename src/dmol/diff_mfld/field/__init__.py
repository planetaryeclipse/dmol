from .base import Field
from .field_types import ScalarField
from .field_ops import (
    _field__add__,
    _field__radd__,
    _field__sub__,
    _field__rsub__,
    _field__mul__,
    _field__rmul__,
    _field__truediv__,
    _field__rtruediv__,
    _field__pow__,
    _MaxField,
)

# prevents a cyclic dependency
Field.__add__ = _field__add__
Field.__radd__ = _field__radd__
Field.__sub__ = _field__sub__
Field.__rsub__ = _field__rsub__
Field.__mul__ = _field__mul__
Field.__rmul__ = _field__rmul__
Field.__truediv__ = _field__truediv__
Field.__rtruediv__ = _field__rtruediv__
Field.__pow__ = _field__pow__

ScalarField.max = _MaxField.create_max


__all__ = ["Field"]
