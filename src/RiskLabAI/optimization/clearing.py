"""Proportional clearing for a network with an outside creditor."""

import numpy as np
from scipy.optimize import linprog


def eisenberg_noe_clearing(
    external_assets: np.ndarray,
    nominal_liabilities: np.ndarray,
    relative_liabilities: np.ndarray,
    *,
    atol: float = 1e-8,
) -> np.ndarray:
    """Return the greatest Eisenberg--Noe clearing payment vector.

    ``relative_liabilities[i, j]`` is the fraction of debtor i's total
    liability owed to creditor j. Nonnegative rows sum to at most one;
    missing mass is owed outside the network. All arrays must be finite,
    with nonnegative assets and liabilities and at least one firm.

    Payments solve ``p = minimum(nominal_liabilities,
    external_assets + relative_liabilities.T @ p)``. Maximizing total
    feasible payments with a linear program selects the greatest solution,
    including when the fixed point is not unique. The model has proportional
    payments, no bankruptcy costs and no fire sales.

    ``atol`` is the absolute verification tolerance after dividing monetary
    quantities by the largest nominal liability. It must lie in (0, 1).
    Invalid inputs raise ValueError; failed or unchecked optimization raises
    RuntimeError. Inputs are not modified.
    """
    arrays = []
    for value in (external_assets, nominal_liabilities, relative_liabilities):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Inputs must contain real numeric values.")
        array = np.asarray(raw, dtype=float)
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError("Inputs must be finite and nonnegative.")
        arrays.append(array)
    assets, liabilities, relative = arrays
    if assets.ndim != 1 or assets.size == 0 or liabilities.shape != assets.shape:
        raise ValueError("Assets and liabilities must be equal nonempty vectors.")
    if relative.shape != (assets.size, assets.size):
        raise ValueError(
            "Relative liabilities must be a square matrix matching assets."
        )
    if np.any(relative.sum(axis=1) > 1):
        raise ValueError("Relative-liability rows must sum to at most one.")
    tolerance = np.asarray(atol)
    if tolerance.ndim != 0 or tolerance.dtype.kind not in "iuf":
        raise ValueError("atol must be finite and lie in (0, 1).")
    atol = float(tolerance)
    if not np.isfinite(atol) or not 0 < atol < 1:
        raise ValueError("atol must be finite and lie in (0, 1).")
    scale = float(liabilities.max())
    if scale == 0:
        return np.zeros_like(liabilities)
    upper = liabilities / scale
    resources = np.minimum(assets, liabilities) / scale
    result = linprog(
        -np.ones(assets.size),
        A_ub=np.eye(assets.size) - relative.T,
        b_ub=resources,
        bounds=np.column_stack((np.zeros(assets.size), upper)),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Clearing optimization failed: {result.message}")
    payments = np.asarray(result.x, dtype=float)
    if (
        payments.shape != liabilities.shape
        or not np.all(np.isfinite(payments))
        or np.any(payments < -atol)
        or np.any(payments > upper + atol)
    ):
        raise RuntimeError("Clearing optimization returned invalid payment bounds.")
    payments = np.clip(payments, 0, upper)
    fixed_point = np.minimum(upper, resources + relative.T @ payments)
    if np.max(np.abs(payments - fixed_point)) > atol:
        raise RuntimeError("Clearing optimization failed the fixed-point check.")
    return payments * scale
