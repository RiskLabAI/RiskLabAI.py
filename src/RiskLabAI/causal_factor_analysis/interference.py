"""Design-based exposure contrasts under a supplied interference mapping."""

import numpy as np

from RiskLabAI.utils._validation import real_array

__all__ = ["exposure_mean_contrast"]


def exposure_mean_contrast(
    outcomes, exposures, exposure_probabilities, *, first, second
):
    """Return the Horvitz-Thompson mean contrast first minus second.

    Each row of exposure_probabilities lists the unit's known probabilities
    under the randomized assignment design. exposures contains integer column
    indices for realized exposures. All units need positive probability for
    both contrasted exposures. The mapping must make potential outcomes
    well-defined; graph edges alone do not establish this assumption.
    No variance or confidence interval is returned without joint exposure
    probabilities. Probabilities need not sum to one if other exposure
    categories are omitted, but each row's sum cannot exceed one.
    """
    y = real_array(outcomes, "outcomes", 1)
    probabilities = real_array(exposure_probabilities, "exposure_probabilities", 2)
    labels = np.asarray(exposures)
    if (
        probabilities.shape[0] != len(y)
        or labels.shape != y.shape
        or labels.dtype.kind not in "iu"
        or np.any(labels < 0)
        or np.any(labels >= probabilities.shape[1])
    ):
        raise ValueError("Exposures must be valid integer columns matching outcomes.")
    for value in (first, second):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= value < probabilities.shape[1]
        ):
            raise ValueError("Contrast indices must name probability columns.")
    if (
        first == second
        or np.any(probabilities < 0)
        or np.any(probabilities.sum(axis=1) > 1 + 1e-12)
    ):
        raise ValueError("Require distinct contrasts and valid probability mass.")
    if np.any(probabilities[:, [first, second]] <= 0) or np.any(
        probabilities[np.arange(len(y)), labels] <= 0
    ):
        raise ValueError("The requested exposure contrast lacks positivity.")
    first_scores = (labels == first) * y / probabilities[:, first]
    second_scores = (labels == second) * y / probabilities[:, second]
    if (
        not np.isfinite(first_scores).all()
        or not np.isfinite(second_scores).all()
        or not np.isfinite(first_scores - second_scores).all()
    ):
        raise ValueError("Exposure scores exceed floating-point range.")
    return {
        "first_mean": float(first_scores.mean()),
        "second_mean": float(second_scores.mean()),
        "contrast": float((first_scores - second_scores).mean()),
        "unit_contributions": first_scores - second_scores,
    }
