"""Built-in predictor adapters.

Importing this package triggers the ``@register_predictor`` decorators
below, populating the registry with shipped adapters.

Add new built-in adapters by dropping a new module in this package and
importing it here.
"""

# Importing the modules triggers their @register_predictor decorators.
from . import openspliceai  # noqa: F401
from . import spliceai  # noqa: F401

__all__ = ["openspliceai", "spliceai"]
