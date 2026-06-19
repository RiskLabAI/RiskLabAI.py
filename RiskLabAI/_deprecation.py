"""
Internal helpers for deprecating public names without breaking callers.

Used by the 2.0.0 naming canon (see ``NAMING_CANON_2.0.0.md``): every renamed
function, class, or constant keeps working throughout the 2.0.x series and emits
a :class:`DeprecationWarning` naming its replacement. The aliases are scheduled
for removal in 2.1.0.

Three mechanisms, one per kind of object:

- :func:`deprecated_alias` wraps a renamed **function**.
- :func:`deprecated_class` subclasses a renamed **class**.
- :func:`warn_deprecated_name` is called from a module ``__getattr__`` (PEP 562)
  for renamed **module-level names** such as constants.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

__all__ = ["deprecated_alias", "deprecated_class", "warn_deprecated_name"]

_F = TypeVar("_F", bound=Callable[..., Any])


def deprecated_alias(new_func: _F, old_name: str, *, removed_in: str) -> _F:
    """
    Return a thin wrapper that calls ``new_func`` but warns under the old name.

    Parameters
    ----------
    new_func : Callable
        The renamed (canonical) function.
    old_name : str
        The deprecated public name being kept alive.
    removed_in : str
        Version in which the alias will be removed (e.g. ``"2.1.0"``).

    Returns
    -------
    Callable
        A wrapper with ``__name__`` set to ``old_name`` that emits a
        :class:`DeprecationWarning` and delegates to ``new_func``.
    """

    @functools.wraps(new_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{old_name}() is deprecated and will be removed in {removed_in}; "
            f"use {new_func.__name__}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    wrapper.__doc__ = (
        f"Deprecated alias for :func:`{new_func.__name__}` "
        f"(removed in {removed_in})."
    )
    return wrapper  # type: ignore[return-value]


def deprecated_class(new_cls: type, old_name: str, *, removed_in: str) -> type:
    """
    Return a subclass of ``new_cls`` that warns when instantiated.

    The subclass behaves identically to ``new_cls`` but emits a
    :class:`DeprecationWarning` on construction, naming the replacement.

    Parameters
    ----------
    new_cls : type
        The renamed (canonical) class.
    old_name : str
        The deprecated public class name being kept alive.
    removed_in : str
        Version in which the alias will be removed.
    """

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            f"{old_name} is deprecated and will be removed in {removed_in}; "
            f"use {new_cls.__name__} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        new_cls.__init__(self, *args, **kwargs)

    shim = type(old_name, (new_cls,), {"__init__": __init__})
    shim.__doc__ = (
        f"Deprecated alias for :class:`{new_cls.__name__}` "
        f"(removed in {removed_in})."
    )
    shim.__module__ = new_cls.__module__
    return shim


def warn_deprecated_name(old_name: str, new_name: str, *, removed_in: str) -> None:
    """
    Emit a deprecation warning for a renamed module-level name.

    Call this from a module's ``__getattr__`` (PEP 562) before returning the
    value bound to ``new_name``.

    Parameters
    ----------
    old_name : str
        The deprecated name that was accessed.
    new_name : str
        The canonical replacement name.
    removed_in : str
        Version in which the old name will be removed.
    """
    warnings.warn(
        f"{old_name} is deprecated and will be removed in {removed_in}; "
        f"use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
