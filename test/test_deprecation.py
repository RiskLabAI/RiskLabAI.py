"""
Unit tests for the deprecation helpers in ``RiskLabAI._deprecation``.

These guarantee the shim mechanism (used by the 2.0.0 naming canon) stays
intact: aliases keep working AND emit a ``DeprecationWarning`` naming the
replacement.
"""

import pytest

from RiskLabAI._deprecation import (
    deprecated_alias,
    deprecated_class,
    warn_deprecated_name,
)


def test_deprecated_alias_calls_through_and_warns():
    def new_func(a, b=2):
        """Add two numbers."""
        return a + b

    old_func = deprecated_alias(new_func, "old_func", removed_in="2.1.0")

    with pytest.warns(DeprecationWarning, match="old_func.*2.1.0.*new_func"):
        result = old_func(3, b=4)

    assert result == 7
    assert old_func.__name__ == "old_func"


def test_deprecated_class_is_subclass_and_warns_on_init():
    class New:
        def __init__(self, x):
            self.x = x

    Old = deprecated_class(New, "Old", removed_in="2.1.0")

    assert issubclass(Old, New)
    assert Old is not New

    with pytest.warns(DeprecationWarning, match="Old.*2.1.0.*New"):
        obj = Old(5)

    assert isinstance(obj, New)
    assert obj.x == 5


def test_warn_deprecated_name_emits_warning():
    with pytest.warns(DeprecationWarning, match="OLD_NAME.*2.1.0.*NEW_NAME"):
        warn_deprecated_name("OLD_NAME", "NEW_NAME", removed_in="2.1.0")


def test_deprecated_theta_constant_alias_warns():
    """The non-ASCII θ constants warn (and return the ASCII value) via utils."""
    import RiskLabAI.utils as utils

    with pytest.warns(DeprecationWarning, match="2.1.0"):
        value = utils.CUMULATIVE_θ

    # Identifier changed; the string value is intentionally unchanged.
    assert value == utils.CUMULATIVE_THETA == "Cumulative θ"
