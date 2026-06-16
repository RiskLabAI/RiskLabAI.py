"""Unit tests for the generic component Registry."""

import pytest

from RiskLabAI.core.registry import Registry


def make_registry():
    return Registry("test")


# --------------------------------------------------------------------------- #
# Registration styles
# --------------------------------------------------------------------------- #
def test_register_as_decorator_with_key():
    reg = make_registry()

    @reg.register("widget")
    class Widget:
        pass

    assert reg.get("widget") is Widget
    assert reg.available() == ["widget"]


def test_register_as_bare_decorator_infers_name():
    reg = make_registry()

    @reg.register
    class Gadget:
        pass

    assert reg.get("Gadget") is Gadget


def test_register_direct_call_returns_object():
    reg = make_registry()

    class Thing:
        pass

    returned = reg.register("thing", Thing)
    assert returned is Thing
    assert reg.get("thing") is Thing


def test_register_function_factory():
    reg = make_registry()

    @reg.register("builder")
    def build():
        return [1, 2, 3]

    assert reg.create("builder") == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Lookup behaviour
# --------------------------------------------------------------------------- #
def test_lookup_is_case_insensitive():
    reg = make_registry()
    reg.register("Widget", object)
    assert reg.get("widget") is object
    assert reg.get("WIDGET") is object
    assert "wIdGeT" in reg


def test_aliases_resolve_to_same_component():
    reg = make_registry()

    class Dog:
        pass

    reg.register("dog", Dog, aliases=("doggo", "pup"))
    assert reg.get("doggo") is Dog
    assert reg.get("pup") is Dog
    assert reg.aliases() == {"doggo": "dog", "pup": "dog"}


def test_unknown_key_raises_with_available_listed():
    reg = make_registry()
    reg.register("alpha", object)
    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    message = str(excinfo.value)
    assert "missing" in message
    assert "'alpha'" in message


def test_non_string_key_raises_typeerror():
    reg = make_registry()
    with pytest.raises(TypeError):
        reg.get(123)


# --------------------------------------------------------------------------- #
# Duplicate protection / override
# --------------------------------------------------------------------------- #
def test_duplicate_registration_raises():
    reg = make_registry()
    reg.register("x", object)
    with pytest.raises(KeyError):
        reg.register("x", dict)


def test_override_replaces_and_clears_old_aliases():
    reg = make_registry()
    reg.register("x", object, aliases=("ex",))
    reg.register("x", dict, override=True)
    assert reg.get("x") is dict
    # Old alias must no longer resolve.
    with pytest.raises(KeyError):
        reg.get("ex")


# --------------------------------------------------------------------------- #
# create() and kwarg filtering
# --------------------------------------------------------------------------- #
def test_create_forwards_args_and_kwargs():
    reg = make_registry()

    @reg.register("point")
    class Point:
        def __init__(self, x, y=0):
            self.x, self.y = x, y

    p = reg.create("point", 1, y=2)
    assert (p.x, p.y) == (1, 2)


def test_create_filter_unknown_kwargs_drops_extras():
    reg = make_registry()

    @reg.register("strict")
    class Strict:
        def __init__(self, a):
            self.a = a

    # Without filtering, an unexpected kwarg is an error (typos surface).
    with pytest.raises(TypeError):
        reg.create("strict", a=1, bogus=2)

    # With filtering, the unknown kwarg is dropped (factory-style behaviour).
    obj = reg.create("strict", a=1, bogus=2, filter_unknown_kwargs=True)
    assert obj.a == 1


def test_create_filter_keeps_all_when_var_keyword():
    reg = make_registry()

    @reg.register("flexible")
    class Flexible:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    obj = reg.create("flexible", a=1, b=2, filter_unknown_kwargs=True)
    assert obj.kwargs == {"a": 1, "b": 2}


# --------------------------------------------------------------------------- #
# Lazy registration
# --------------------------------------------------------------------------- #
def test_register_lazy_defers_import_then_resolves():
    reg = make_registry()
    reg.register_lazy("ordered", "collections:OrderedDict")
    assert reg.is_lazy("ordered") is True

    from collections import OrderedDict

    assert reg.get("ordered") is OrderedDict
    # After resolution the entry is no longer lazy.
    assert reg.is_lazy("ordered") is False


def test_register_lazy_bad_target_raises_on_use():
    reg = make_registry()
    reg.register_lazy("bad", "collections-no-colon")
    with pytest.raises(ValueError):
        reg.get("bad")


# --------------------------------------------------------------------------- #
# Introspection / mapping protocol / metadata / unregister
# --------------------------------------------------------------------------- #
def test_mapping_protocol_and_sorting():
    reg = make_registry()
    reg.register("b", object)
    reg.register("a", dict)
    assert reg.available() == ["a", "b"]
    assert list(reg) == ["a", "b"]
    assert len(reg) == 2
    assert reg["a"] is dict


def test_metadata_round_trips():
    reg = make_registry()
    reg.register("m", object, metadata={"family": "demo", "page": 42})
    assert reg.metadata("m") == {"family": "demo", "page": 42}


def test_unregister_removes_key_and_aliases():
    reg = make_registry()
    reg.register("z", object, aliases=("zee",))
    reg.unregister("z")
    assert "z" not in reg
    assert "zee" not in reg
    assert reg.available() == []
