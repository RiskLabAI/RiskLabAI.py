"""Probability uncertainty over supplied priors, separate from return variance."""

import numpy as np
from scipy.special import ndtr

from RiskLabAI.utils._validation import probability_rows, real_array, real_scalar

__all__ = ["probability_ambiguity", "normal_prior_bin_probabilities"]


def probability_ambiguity(probabilities, *, prior_weights=None, bin_width=None):
    """Return sum of prior-mean bin probability times its population variance.

    Rows are priors, columns are mutually exclusive exhaustive outcome bins.
    Explicit prior weights must sum to one; the default weights are uniform.
    Variance uses the finite prior distribution, with no sample correction.
    If supplied, bin_width must be in (0, 1); scaled is unscaled/[w*(1-w)].
    This implements the discrete Brenner-Izhakian supplied-prior measure,
    not their raw intraday probability-estimation pipeline.
    """
    probabilities = probability_rows(probabilities, "probabilities")
    weights = (
        np.full(len(probabilities), 1 / len(probabilities))
        if prior_weights is None
        else probability_rows(prior_weights, "prior_weights", 1)
    )
    if weights.shape != (len(probabilities),):
        raise ValueError("There must be one weight per prior.")
    mean = weights @ probabilities
    variance = weights @ ((probabilities - mean) ** 2)
    value = float(mean @ variance)
    width = (
        None
        if bin_width is None
        else real_scalar(bin_width, "bin_width", minimum=0, strict=True)
    )
    if width is not None and width >= 1:
        raise ValueError("bin_width must be less than one.")
    return {
        "unscaled": value,
        "scaled": None if width is None else value / (width * (1 - width)),
        "mean_probabilities": mean,
        "probability_variances": variance,
        "prior_weights": weights,
        "bin_width": width,
    }


def normal_prior_bin_probabilities(means, standard_deviations, *, edges=None):
    """Convert supplied normal priors to exhaustive bin probabilities.

    Default finite edges are -0.06 through 0.06 in steps of 0.002.
    Two tail bins are included. Means and standard deviations must use
    the same return units as the edges. Standard deviations are positive.
    """
    means = real_array(means, "means", 1)
    scales = real_array(standard_deviations, "standard_deviations", 1)
    edges = (
        np.linspace(-0.06, 0.06, 61) if edges is None else real_array(edges, "edges", 1)
    )
    if (
        scales.shape != means.shape
        or np.any(scales <= 0)
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "Scales must match means and be positive; edges must increase."
        )
    z = (edges[None, :] - means[:, None]) / scales[:, None]
    interior = np.where(
        z[:, :-1] >= 0,
        ndtr(-z[:, :-1]) - ndtr(-z[:, 1:]),
        ndtr(z[:, 1:]) - ndtr(z[:, :-1]),
    )
    return np.column_stack((ndtr(z[:, 0]), interior, ndtr(-z[:, -1])))
