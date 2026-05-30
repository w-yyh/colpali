"""ColPali engine package.

Model classes are intentionally loaded lazily so ColQwen2-only experiments do
not import every optional model family at startup.
"""

from importlib import import_module

from .models import __all__ as _MODEL_NAMES

__all__ = list(_MODEL_NAMES)


def __getattr__(name: str):
    if name in _MODEL_NAMES:
        return getattr(import_module("colpali_engine.models"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
