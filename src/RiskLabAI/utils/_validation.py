"""Small input checks shared by numerical methods."""

import numpy as np


def real_array(value, name, ndim=None, *, empty=False):
    """Return an independent finite float array of the requested dimension."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f"{name} must be a real numeric array of dimension {ndim}.")
    result = np.array(raw, dtype=float, copy=True)
    if (not empty and result.size == 0) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values and cannot be empty.")
    return result


def real_scalar(value, name, *, minimum=None, strict=False):
    """Validate a finite scalar with an optional lower bound."""
    result = float(real_array(value, name, 0))
    if minimum is not None and (result <= minimum if strict else result < minimum):
        raise ValueError(f"{name} is below its allowed bound.")
    return result


def positive_integer(value, name):
    """Reject booleans and nonpositive or nonintegral counts."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def probability_rows(value, name, ndim=2):
    """Validate explicit probability vectors without clipping or renormalizing."""
    result = real_array(value, name, ndim)
    if np.any((result < 0) | (result > 1)) or not np.allclose(
        result.sum(axis=-1), 1, atol=1e-12, rtol=1e-12
    ):
        raise ValueError(f"{name} must have nonnegative rows summing to one.")
    return result
