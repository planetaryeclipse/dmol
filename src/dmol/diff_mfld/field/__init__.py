from .base import Field
from .field_ops import _field__add__, _field__sub__, _field__mul__, _field__rmul__

# prevents a cyclic dependency
Field.__add__ = _field__add__
Field.__sub__ = _field__sub__
Field.__mul__ = _field__mul__
Field.__rmul__ = _field__rmul__

__all__ = ["Field"]
