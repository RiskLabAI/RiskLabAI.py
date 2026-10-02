"""Finite vector majorization and checked doubly stochastic representations."""

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix


def _absolute_tolerance(atol: float) -> float:
    if (
        isinstance(atol, (bool, np.bool_))
        or not isinstance(atol, (int, float, np.integer, np.floating))
        or not np.isfinite(atol)
        or not 0 < atol < 1
    ):
        raise ValueError(
            "atol must be a finite real number strictly between zero and one"
        )
    return float(atol)


def _real_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(float, copy=True)


def _vectors(target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = _real_array(target, "target")
    right = _real_array(source, "source")
    if left.ndim != 1 or right.shape != left.shape or left.size == 0:
        raise ValueError("target and source must be nonempty vectors of equal length")
    return left, right


def is_doubly_stochastic(matrix: np.ndarray, *, atol: float = 1e-10) -> bool:
    """Check nonnegativity and unit row and column sums within absolute tolerance.

    Return False for a non-square, empty, nonfinite or non-real matrix.
    Entries as small as ``-atol`` are accepted as numerical roundoff.
    ``atol`` must be finite and strictly between zero and one. No matrix
    normalization or input mutation is performed.
    """
    tolerance = _absolute_tolerance(atol)
    try:
        values = _real_array(matrix, "matrix")
    except (ValueError, TypeError):
        return False
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != values.shape[1]:
        return False
    return bool(
        np.all(values >= -tolerance)
        and np.all(np.abs(values.sum(axis=0) - 1.0) <= tolerance)
        and np.all(np.abs(values.sum(axis=1) - 1.0) <= tolerance)
    )


def is_majorized(
    target: np.ndarray, source: np.ndarray, *, atol: float = 1e-10
) -> bool:
    """Return whether ``target`` is majorized by ``source``.

    Sort both finite real vectors in descending order. Every proper partial
    sum of target must be no greater than the corresponding source sum, and
    their totals must agree. All comparisons use the absolute tolerance
    ``atol``; negative entries are allowed. Unequal dimensions and nonfinite
    inputs raise ValueError. This is ordinary equal-total vector majorization.

    Examples
    --------
    >>> is_majorized([0.5, 0.5], [1.0, 0.0])
    True
    >>> is_majorized([1.0, 0.0], [0.5, 0.5])
    False
    """
    tolerance = _absolute_tolerance(atol)
    left, right = _vectors(target, source)
    left_sums = np.cumsum(np.sort(left)[::-1], dtype=np.longdouble)
    right_sums = np.cumsum(np.sort(right)[::-1], dtype=np.longdouble)
    if not np.all(np.isfinite(left_sums)) or not np.all(np.isfinite(right_sums)):
        raise ValueError("partial sums exceed the supported numerical range")
    return bool(
        abs(left_sums[-1] - right_sums[-1]) <= tolerance
        and np.all(left_sums[:-1] <= right_sums[:-1] + tolerance)
    )


def majorization_matrix(
    target: np.ndarray, source: np.ndarray, *, atol: float = 1e-10
) -> np.ndarray:
    """Construct a doubly stochastic witness ``D @ source == target``.

    Solve the finite linear feasibility problem with SciPy's HiGHS backend.
    Validate row sums, column sums, nonnegativity and the mapping residual
    before returning the matrix. All checks use absolute tolerance ``atol``.
    The witness need not be unique and its entries may vary across solvers.
    Raise ValueError when the inputs fail the majorization check and
    RuntimeError when a numerically verified witness cannot be obtained.

    Examples
    --------
    >>> source = np.array([1.0, 0.0])
    >>> target = np.array([0.5, 0.5])
    >>> np.allclose(majorization_matrix(target, source) @ source, target)
    True
    """
    tolerance = _absolute_tolerance(atol)
    left, right = _vectors(target, source)
    if not is_majorized(left, right, atol=tolerance):
        raise ValueError("target is not majorized by source within atol")
    size = left.size
    if np.all(np.abs(left - right) <= tolerance):
        return np.eye(size)
    scale = max(1.0, float(np.max(np.abs(right))), float(np.max(np.abs(left))))
    columns = np.arange(size * size)
    row_indices = np.repeat(np.arange(size), size)
    column_indices = np.tile(np.arange(size), size)
    constraints = coo_matrix(
        (
            np.concatenate(
                [
                    np.ones(size * size),
                    np.ones(size * size),
                    np.tile(right / scale, size),
                ]
            ),
            (
                np.concatenate(
                    [row_indices, size + column_indices, 2 * size + row_indices]
                ),
                np.tile(columns, 3),
            ),
        ),
        shape=(3 * size, size * size),
    ).tocsr()
    result = linprog(
        np.zeros(size * size),
        A_eq=constraints,
        b_eq=np.concatenate([np.ones(2 * size), left / scale]),
        bounds=(0.0, 1.0),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"No verified majorization witness: {result.message}")
    matrix = result.x.reshape(size, size)
    if not is_doubly_stochastic(matrix, atol=tolerance) or not np.all(
        np.abs(matrix @ right - left) <= tolerance
    ):
        raise RuntimeError(
            "Majorization witness exceeds the requested residual tolerance"
        )
    return matrix


def birkhoff_von_neumann_decomposition(
    matrix: np.ndarray, *, atol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Represent a doubly stochastic matrix as weighted permutation matrices.

    Return ``(weights, permutations)``. Component k selects column
    ``permutations[k, i]`` in row i, so its matrix is
    ``np.eye(n)[permutations[k]]``. The weighted sum reconstructs the input
    within absolute tolerance ``atol``. Weights are positive and sum to one
    within that tolerance. The decomposition is generally nonunique.

    SciPy finds a matching on the positive residual support at each step.
    Inputs are neither normalized nor mutated. Tiny negative entries within
    tolerance are treated as roundoff, and the final result is checked
    against the original input. Raise ValueError for an invalid input and
    RuntimeError if a checked decomposition cannot be obtained numerically.

    Examples
    --------
    >>> matrix = np.array([[0.25, 0.75], [0.75, 0.25]])
    >>> weights, permutations = birkhoff_von_neumann_decomposition(matrix)
    >>> np.allclose(sum(w * np.eye(2)[p] for w, p in zip(weights, permutations)), matrix)
    True
    """
    tolerance = _absolute_tolerance(atol)
    if not is_doubly_stochastic(matrix, atol=tolerance):
        raise ValueError("matrix must be doubly stochastic within atol")
    original = _real_array(matrix, "matrix")
    residual = np.maximum(original, 0.0)
    reconstruction = np.zeros_like(original)
    weights = []
    permutations = []
    mass = 0.0
    for _ in range(original.size):
        if abs(mass - 1.0) <= tolerance and np.max(np.abs(residual)) <= tolerance:
            break
        try:
            rows, columns = linear_sum_assignment(np.where(residual > 0.0, 0.0, np.inf))
        except ValueError as error:
            raise RuntimeError(
                "Residual support has no complete positive matching"
            ) from error
        weight = float(np.min(residual[rows, columns]))
        if weight <= 0.0:
            raise RuntimeError("Decomposition did not make positive progress")
        weights.append(weight)
        permutations.append(columns)
        residual[rows, columns] -= weight
        reconstruction[rows, columns] += weight
        mass += weight
    if (
        abs(mass - 1.0) > tolerance
        or np.max(np.abs(reconstruction - original)) > tolerance
    ):
        raise RuntimeError("Decomposition exceeds the requested residual tolerance")
    return np.asarray(weights), np.asarray(permutations, dtype=np.intp)
