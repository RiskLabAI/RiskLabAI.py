"""
A small, dependency-free component registry.

Several sub-packages in RiskLabAI already implement the same pattern by hand:
a string key is mapped to a class, and a factory instantiates the right class
from that key (``controller/bars_initializer.py``,
``backtest/validation/cross_validator_factory.py``,
``features/feature_importance/feature_importance_factory.py``). Each does it
slightly differently (different casing rules, different error behaviour, a
hand-written ``dict`` in three places).

``Registry`` factors that pattern into one reusable, well-tested object so that:

* every model family discovers and instantiates its components the same way;
* **new** models (e.g. an extension proposed in a RiskLabAI methods paper) can be
  added with a single ``@registry.register("my_model")`` decorator, without
  editing a central ``dict`` — the open/closed principle in practice;
* lookups fail loudly with a helpful message listing the valid keys, instead of
  silently returning ``None`` or an empty result.

The registry is intentionally tiny and imports nothing heavy, so importing it
(and the :mod:`RiskLabAI.core` package) stays cheap. Components may be registered
*lazily* by a ``"module.path:attribute"`` string, so registering the built-in
catalogue does not import pandas/numba/torch until a component is actually
created.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
)

logger = logging.getLogger(__name__)

__all__ = ["Registry"]

# A factory is anything callable that returns a component instance: usually a
# class, but a plain function works too.
Factory = Callable[..., Any]


class _Entry:
    """Internal record for one registered component (eager or lazy)."""

    __slots__ = ("key", "_obj", "_lazy_target", "metadata")

    def __init__(
        self,
        key: str,
        obj: Factory | None = None,
        lazy_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.key = key
        self._obj = obj
        self._lazy_target = lazy_target
        self.metadata: dict[str, Any] = dict(metadata or {})

    @property
    def is_lazy(self) -> bool:
        """True if the component has not been imported/resolved yet."""
        return self._obj is None

    def resolve(self) -> Factory:
        """Return the factory, importing it on first use if registered lazily."""
        if self._obj is None:
            if not self._lazy_target:
                raise RuntimeError(
                    f"Registry entry {self.key!r} has neither an object nor a "
                    f"lazy import target."
                )
            module_path, sep, attribute = self._lazy_target.partition(":")
            if not sep:
                raise ValueError(
                    f"Lazy target {self._lazy_target!r} for {self.key!r} must be "
                    f"of the form 'package.module:attribute'."
                )
            module = importlib.import_module(module_path)
            self._obj = getattr(module, attribute)
        return self._obj


class Registry:
    """
    A name -> component factory registry with case-insensitive lookup.

    Parameters
    ----------
    name : str
        Human-readable name of the registry (used in error messages),
        e.g. ``"bars"`` or ``"cross_validators"``.
    base : type, optional
        An optional expected base class / interface for registered components.
        It is **not** enforced (so existing free functions and duck-typed
        components keep working), but it is recorded for documentation and can
        be inspected via :attr:`base`.

    Examples
    --------
    >>> from RiskLabAI.core.registry import Registry
    >>> animals = Registry("animals")
    >>> @animals.register("dog", aliases=("doggo",))
    ... class Dog:
    ...     def speak(self) -> str:
    ...         return "woof"
    >>> animals.available()
    ['dog']
    >>> animals.create("DOG").speak()      # case-insensitive
    'woof'
    >>> animals.create("doggo").speak()    # alias
    'woof'
    """

    def __init__(self, name: str, *, base: type | None = None) -> None:
        self.name = name
        self.base = base
        # canonical key -> entry
        self._entries: dict[str, _Entry] = {}
        # lower-cased key OR alias -> canonical key
        self._index: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(
        self,
        key: Any = None,
        obj: Factory | None = None,
        *,
        aliases: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        override: bool = False,
    ) -> Any:
        """
        Register a component. Usable as a decorator or as a direct call.

        Decorator forms::

            @reg.register("my_model")
            class MyModel: ...

            @reg.register                       # key inferred from class name
            class MyModel: ...

        Direct form::

            reg.register("my_model", MyModel)

        Parameters
        ----------
        key : str or callable, optional
            The lookup key. When used as a bare decorator (``@reg.register``)
            this is the class/function being decorated and the key is taken from
            its ``__name__``.
        obj : callable, optional
            The factory (class or function). If given, this is a direct
            registration and ``obj`` is returned unchanged.
        aliases : tuple of str, optional
            Additional names that resolve to the same component.
        metadata : dict, optional
            Arbitrary metadata stored alongside the entry (e.g. a description,
            an AFML page reference, the family it belongs to).
        override : bool, default False
            Allow replacing an existing key. Otherwise a duplicate raises
            :class:`KeyError`, which protects against accidental shadowing.

        Returns
        -------
        The decorated object (decorator forms) or ``obj`` (direct form).
        """
        # Bare decorator: @reg.register  (key is the class/function itself)
        if obj is None and callable(key) and not isinstance(key, str):
            target = key
            self._register(
                target.__name__,
                target,
                aliases=aliases,
                metadata=metadata,
                override=override,
            )
            return target

        # Direct call: reg.register("name", obj)
        if obj is not None:
            self._register(
                key,
                obj,
                aliases=aliases,
                metadata=metadata,
                override=override,
            )
            return obj

        # Decorator factory: @reg.register("name")
        def decorator(target: Factory) -> Factory:
            resolved_key = key if isinstance(key, str) else target.__name__
            self._register(
                resolved_key,
                target,
                aliases=aliases,
                metadata=metadata,
                override=override,
            )
            return target

        return decorator

    def register_lazy(
        self,
        key: str,
        target: str,
        *,
        aliases: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        override: bool = False,
    ) -> None:
        """
        Register a component by import path, deferring the import to first use.

        Parameters
        ----------
        key : str
            The lookup key.
        target : str
            Import path of the form ``"package.module:attribute"``.
        aliases, metadata, override
            See :meth:`register`.
        """
        self._register(
            key,
            None,
            lazy_target=target,
            aliases=aliases,
            metadata=metadata,
            override=override,
        )

    def _register(
        self,
        key: str,
        obj: Factory | None,
        *,
        lazy_target: str | None = None,
        aliases: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        override: bool = False,
    ) -> None:
        if not isinstance(key, str) or not key:
            raise TypeError(f"Registry key must be a non-empty string, got {key!r}.")
        lower = key.lower()
        if lower in self._index and not override:
            existing = self._index[lower]
            raise KeyError(
                f"{self.name!r} registry already has a component under "
                f"{key!r} (canonical key {existing!r}). "
                f"Pass override=True to replace it."
            )
        # On override, drop any stale aliases pointing at the old canonical key.
        if override and lower in self._index:
            old_canonical = self._index[lower]
            self._entries.pop(old_canonical, None)
            self._index = {k: v for k, v in self._index.items() if v != old_canonical}

        self._entries[key] = _Entry(
            key,
            obj=obj,
            lazy_target=lazy_target,
            metadata=metadata,
        )
        self._index[lower] = key
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                raise TypeError(f"Alias must be a non-empty string, got {alias!r}.")
            self._index[alias.lower()] = key
        logger.debug("Registered %r in %r registry.", key, self.name)

    def unregister(self, key: str) -> None:
        """Remove a component and all of its aliases. Raises if absent."""
        entry = self._lookup(key)
        canonical = entry.key
        self._entries.pop(canonical, None)
        self._index = {k: v for k, v in self._index.items() if v != canonical}

    # ------------------------------------------------------------------ #
    # Lookup & construction
    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Factory:
        """Return the registered factory (class/callable) for ``key``."""
        return self._lookup(key).resolve()

    def create(
        self,
        key: str,
        *args: Any,
        filter_unknown_kwargs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Instantiate the component registered under ``key``.

        Parameters
        ----------
        key : str
            The lookup key (case-insensitive; aliases accepted).
        *args, **kwargs
            Forwarded to the component's constructor.
        filter_unknown_kwargs : bool, default False
            If True, silently drop keyword arguments the constructor does not
            accept (unless it accepts ``**kwargs``). This mirrors the behaviour
            of the existing cross-validator / feature-importance factories,
            which let callers pass a shared bag of options. Off by default so
            that typos surface as clear ``TypeError``s.
        """
        factory = self.get(key)
        if filter_unknown_kwargs:
            kwargs = _filter_kwargs(factory, kwargs)
        return factory(*args, **kwargs)

    def metadata(self, key: str) -> dict[str, Any]:
        """Return the metadata dict registered alongside ``key``."""
        return dict(self._lookup(key).metadata)

    def is_lazy(self, key: str) -> bool:
        """True if ``key`` is registered lazily and not yet imported."""
        return self._lookup(key).is_lazy

    def _lookup(self, key: str) -> _Entry:
        try:
            canonical = self._index[key.lower()]
        except AttributeError:
            raise TypeError(
                f"Registry key must be a string, got {type(key).__name__}."
            ) from None
        except KeyError:
            raise KeyError(self._not_found_message(key)) from None
        return self._entries[canonical]

    def _not_found_message(self, key: str) -> str:
        available = self.available()
        listing = ", ".join(repr(k) for k in available) if available else "(none)"
        return (
            f"{key!r} is not registered in the {self.name!r} registry. "
            f"Available components: {listing}."
        )

    # ------------------------------------------------------------------ #
    # Introspection / mapping protocol
    # ------------------------------------------------------------------ #
    def available(self) -> list[str]:
        """Sorted list of canonical keys."""
        return sorted(self._entries.keys())

    def keys(self) -> list[str]:
        """Alias for :meth:`available` (mapping-like access)."""
        return self.available()

    def aliases(self) -> dict[str, str]:
        """Mapping of alias -> canonical key (excludes canonical keys themselves)."""
        return {
            alias: canonical
            for alias, canonical in self._index.items()
            if alias != canonical.lower()
        }

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._index

    def __getitem__(self, key: str) -> Factory:
        return self.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.available())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Registry {self.name!r}: {len(self)} components>"


def _filter_kwargs(factory: Factory, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Drop keyword arguments the factory's signature does not accept.

    If the signature contains ``**kwargs`` (VAR_KEYWORD), everything is kept.
    """
    target = factory.__init__ if inspect.isclass(factory) else factory
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover - builtins without sig
        return kwargs
    params = signature.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs
    accepted = {p.name for p in params}
    return {k: v for k, v in kwargs.items() if k in accepted}
