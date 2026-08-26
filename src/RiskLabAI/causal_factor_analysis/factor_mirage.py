"""Closed-form analytics for specification errors in factor investing.

The functions implement the standardized linear-Gaussian results in Lopez de
Prado and Zoonekynd (2026), Equations 2, 5-19 and Appendices D-E. Factor
returns condition on the supplied exposures. Forecast returns integrate over
the exposure distribution assumed by the article.

The generalized coefficient functions retain the disturbance variances from
the population normal equations. The released standardized functions remain
the unit-variance special cases with unchanged signatures.

Appendix F is not exposed as a separate parameter-shift model because its
equations contain only the training-regime parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from numbers import Integral, Real
from typing import Optional

__all__ = [
    "ColliderCoefficients",
    "ColliderDiagnostics",
    "StrategyPerformance",
    "collider_factor_return",
    "collider_forecast_return",
    "collider_model_diagnostics",
    "collider_overcontrolled_coefficients",
    "confounder_factor_return",
    "confounder_forecast_return",
    "confounder_undercontrolled_coefficient",
    "generalized_collider_overcontrolled_coefficients",
    "generalized_confounder_undercontrolled_coefficient",
]


@dataclass(frozen=True)
class StrategyPerformance:
    """Expected returns from the correct and misspecified strategies."""

    correct: float
    misspecified: float


@dataclass(frozen=True)
class ColliderCoefficients:
    """Population coefficients after conditioning on the collider."""

    beta_hat: float
    theta_hat: float


@dataclass(frozen=True)
class ColliderDiagnostics:
    """Population model-selection diagnostics for collider conditioning.

    The t-statistic fields are the expected signals in Appendix D. They are
    not p-values. In particular, the article assigns different degrees of
    freedom to the correct and overcontrolled regressions.
    """

    residual_variance: float
    outcome_variance: float
    correct_r_squared: float
    overcontrolled_r_squared: float
    correct_adjusted_r_squared: float
    overcontrolled_adjusted_r_squared: float
    correct_beta_variance: float
    overcontrolled_beta_variance: float
    collider_coefficient_variance: float
    correct_beta_t_statistic: float
    overcontrolled_beta_t_statistic: float
    collider_t_statistic: float
    adjusted_r_squared_prefers_overcontrolled: bool
    absolute_beta_t_prefers_overcontrolled: bool


def _as_finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a representable real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_sample_size(value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("n_observations must be an integer")
    result = int(value)
    if result <= 3:
        raise ValueError("n_observations must be greater than 3")
    return result


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


def _positive_fraction(name: str, value: Real) -> Fraction:
    result = _as_finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return _fraction(result)


def _as_result(name: str, value: Fraction) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} cannot be represented as a finite float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} cannot be represented as a finite float")
    return result


def _as_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _coefficient_times_square_root(
    name: str, coefficient: Fraction, radicand: Fraction
) -> float:
    if coefficient == 0:
        return 0.0
    try:
        with localcontext() as context:
            context.prec = 80
            value = _as_decimal(coefficient) * _as_decimal(radicand).sqrt()
        result = float(value)
    except (ArithmeticError, OverflowError, ValueError) as error:
        raise ValueError(f"{name} cannot be represented as a finite float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} cannot be represented as a finite float")
    return result


def _confounder_coefficient(
    beta: Fraction, gamma: Fraction, delta: Fraction
) -> Fraction:
    return beta + gamma * delta / (1 + delta * delta)


def _collider_coefficients(
    beta: Fraction, gamma: Fraction, delta: Fraction
) -> tuple[Fraction, Fraction]:
    denominator = 1 + gamma * gamma
    return (beta - delta * gamma) / denominator, gamma / denominator


def confounder_undercontrolled_coefficient(
    beta: Real, gamma: Real, delta: Real
) -> float:
    r"""Return the coefficient obtained when the confounder is omitted.

    For ``X = delta * Z + v`` and ``Y = beta * X + gamma * Z + u``, the
    undercontrolled regression coefficient is
    ``beta + gamma * delta / (1 + delta**2)`` (Equation 2).
    """

    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    delta_value = _fraction(_as_finite_real("delta", delta))
    return _as_result(
        "undercontrolled coefficient",
        _confounder_coefficient(beta_value, gamma_value, delta_value),
    )


def generalized_confounder_undercontrolled_coefficient(
    beta: Real,
    gamma: Real,
    delta: Real,
    *,
    confounder_variance: Real,
    exposure_noise_variance: Real,
) -> float:
    r"""Return the omitted-confounder coefficient for general variances.

    For ``X = delta * Z + v`` and ``Y = beta * X + gamma * Z + u``, with
    independent centered disturbances, the population coefficient is

    ``beta + delta * gamma * Var(Z) / (delta**2 * Var(Z) + Var(v))``.

    Both variances must be strictly positive. Setting both to one reproduces
    :func:`confounder_undercontrolled_coefficient` exactly.
    """

    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    delta_value = _fraction(_as_finite_real("delta", delta))
    confounder_variance_value = _positive_fraction(
        "confounder_variance", confounder_variance
    )
    exposure_noise_variance_value = _positive_fraction(
        "exposure_noise_variance", exposure_noise_variance
    )
    denominator = (
        delta_value * delta_value * confounder_variance_value
        + exposure_noise_variance_value
    )
    coefficient = beta_value + (
        delta_value * gamma_value * confounder_variance_value / denominator
    )
    return _as_result("generalized undercontrolled coefficient", coefficient)


def confounder_factor_return(
    x: Real,
    z: Real,
    beta: Real,
    gamma: Real,
    delta_estimated: Real,
    *,
    delta_realized: Optional[Real] = None,  # noqa: UP045, Python 3.9 API
) -> StrategyPerformance:
    r"""Return correct and undercontrolled factor-strategy performance.

    With one correlation regime, omit ``delta_realized`` to evaluate Equations
    5 and 7. Supply it to evaluate the train-to-execution shift in Appendix E,
    where ``delta_estimated`` is the training value and ``delta_realized`` is
    the execution value.
    """

    x_value = _fraction(_as_finite_real("x", x))
    z_value = _fraction(_as_finite_real("z", z))
    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    estimated = _fraction(_as_finite_real("delta_estimated", delta_estimated))
    realized = (
        estimated
        if delta_realized is None
        else _fraction(_as_finite_real("delta_realized", delta_realized))
    )

    correct_signal = x_value * beta_value + z_value * gamma_value
    estimated_coefficient = _confounder_coefficient(beta_value, gamma_value, estimated)
    realized_coefficient = _confounder_coefficient(beta_value, gamma_value, realized)
    correct = correct_signal * correct_signal
    misspecified = x_value * x_value * estimated_coefficient * realized_coefficient
    return StrategyPerformance(
        correct=_as_result("correct factor return", correct),
        misspecified=_as_result("undercontrolled factor return", misspecified),
    )


def confounder_forecast_return(
    beta: Real,
    gamma: Real,
    delta_estimated: Real,
    *,
    delta_realized: Optional[Real] = None,  # noqa: UP045, Python 3.9 API
) -> StrategyPerformance:
    r"""Return correct and undercontrolled forecasting performance.

    Omitting ``delta_realized`` evaluates Equations 8 and 9. Supplying it
    evaluates Equations E-3 and E-4 after a change in the confounder link.
    """

    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    estimated = _fraction(_as_finite_real("delta_estimated", delta_estimated))
    realized = (
        estimated
        if delta_realized is None
        else _fraction(_as_finite_real("delta_realized", delta_realized))
    )

    estimated_coefficient = _confounder_coefficient(beta_value, gamma_value, estimated)
    realized_coefficient = _confounder_coefficient(beta_value, gamma_value, realized)
    second_signal = beta_value * realized + gamma_value
    correct = beta_value * beta_value + second_signal * second_signal
    misspecified = (
        (1 + realized * realized) * estimated_coefficient * realized_coefficient
    )
    return StrategyPerformance(
        correct=_as_result("correct forecast return", correct),
        misspecified=_as_result("undercontrolled forecast return", misspecified),
    )


def collider_overcontrolled_coefficients(
    beta: Real, gamma: Real, delta: Real
) -> ColliderCoefficients:
    r"""Return the two coefficients after conditioning on the collider.

    The coefficient on ``X`` is ``(beta - delta*gamma)/(1 + gamma**2)`` and
    the coefficient on ``Z`` is ``gamma/(1 + gamma**2)`` (Equations 11-12).
    """

    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    delta_value = _fraction(_as_finite_real("delta", delta))
    beta_hat, theta_hat = _collider_coefficients(beta_value, gamma_value, delta_value)
    return ColliderCoefficients(
        beta_hat=_as_result("overcontrolled beta coefficient", beta_hat),
        theta_hat=_as_result("collider coefficient", theta_hat),
    )


def generalized_collider_overcontrolled_coefficients(
    beta: Real,
    gamma: Real,
    delta: Real,
    *,
    outcome_noise_variance: Real,
    collider_noise_variance: Real,
) -> ColliderCoefficients:
    r"""Return collider-conditioned coefficients for general variances.

    For ``Y = beta * X + u`` and ``Z = gamma * Y + delta * X + v``, with
    independent centered disturbances, the population coefficients are

    ``beta_hat = (beta * Var(v) - delta * gamma * Var(u)) /
    (Var(v) + gamma**2 * Var(u))`` and
    ``theta_hat = gamma * Var(u) / (Var(v) + gamma**2 * Var(u))``.

    Both variances must be strictly positive. Setting both to one reproduces
    :func:`collider_overcontrolled_coefficients` exactly.
    """

    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    delta_value = _fraction(_as_finite_real("delta", delta))
    outcome_noise_variance_value = _positive_fraction(
        "outcome_noise_variance", outcome_noise_variance
    )
    collider_noise_variance_value = _positive_fraction(
        "collider_noise_variance", collider_noise_variance
    )
    denominator = (
        collider_noise_variance_value
        + gamma_value * gamma_value * outcome_noise_variance_value
    )
    beta_hat = (
        beta_value * collider_noise_variance_value
        - delta_value * gamma_value * outcome_noise_variance_value
    ) / denominator
    theta_hat = gamma_value * outcome_noise_variance_value / denominator
    return ColliderCoefficients(
        beta_hat=_as_result("generalized overcontrolled beta coefficient", beta_hat),
        theta_hat=_as_result("generalized collider coefficient", theta_hat),
    )


def collider_factor_return(
    x: Real,
    collider_proxy: Real,
    beta: Real,
    gamma: Real,
    delta: Real,
) -> StrategyPerformance:
    r"""Return correct and overcontrolled factor-strategy performance.

    ``collider_proxy`` is the article's pre-outcome proxy ``Z tilde``. It is
    not the contemporaneous collider, which contains the outcome. The return
    maps to Equations 16 and 17.
    """

    x_value = _fraction(_as_finite_real("x", x))
    proxy_value = _fraction(_as_finite_real("collider_proxy", collider_proxy))
    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    delta_value = _fraction(_as_finite_real("delta", delta))

    beta_x = beta_value * x_value
    denominator = 1 + gamma_value * gamma_value
    overcontrolled_signal = (
        beta_value - delta_value * gamma_value
    ) * x_value + gamma_value * proxy_value
    correct = beta_x * beta_x
    misspecified = beta_x * overcontrolled_signal / denominator
    return StrategyPerformance(
        correct=_as_result("correct collider factor return", correct),
        misspecified=_as_result("overcontrolled factor return", misspecified),
    )


def collider_forecast_return(
    beta: Real, gamma: Real, delta: Real
) -> StrategyPerformance:
    r"""Return correct and overcontrolled forecasting performance.

    The two values are ``beta**2`` and
    ``beta * (beta - gamma*delta) / (1 + gamma**2)`` (Equations 18-19).
    """

    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    delta_value = _fraction(_as_finite_real("delta", delta))
    denominator = 1 + gamma_value * gamma_value
    correct = beta_value * beta_value
    misspecified = beta_value * (beta_value - gamma_value * delta_value) / denominator
    return StrategyPerformance(
        correct=_as_result("correct collider forecast return", correct),
        misspecified=_as_result("overcontrolled forecast return", misspecified),
    )


def collider_model_diagnostics(
    beta: Real,
    gamma: Real,
    delta: Real,
    n_observations: Integral,
) -> ColliderDiagnostics:
    r"""Return the Appendix D diagnostics for conditioning on a collider.

    ``n_observations`` must be an integer greater than three. The returned
    booleans implement the strict adjusted-R-squared and absolute expected
    beta t-statistic comparisons. The latter is not an exact p-value test.
    """

    beta_value = _fraction(_as_finite_real("beta", beta))
    gamma_value = _fraction(_as_finite_real("gamma", gamma))
    delta_value = _fraction(_as_finite_real("delta", delta))
    n_integer = _as_sample_size(n_observations)
    n_value = Fraction(n_integer)

    gamma_denominator = 1 + gamma_value * gamma_value
    outcome_variance = 1 + beta_value * beta_value
    residual_variance = 1 / gamma_denominator
    correct_r_squared = 1 - 1 / outcome_variance
    overcontrolled_r_squared = 1 - 1 / (gamma_denominator * outcome_variance)
    correct_adjusted_r_squared = 1 - (
        Fraction(n_integer - 1, n_integer - 2) / outcome_variance
    )
    overcontrolled_adjusted_r_squared = 1 - (
        Fraction(n_integer - 1, n_integer - 3) / (gamma_denominator * outcome_variance)
    )

    diagnostic_sum = (
        (beta_value * gamma_value + delta_value) ** 2 + gamma_value * gamma_value + 1
    )
    correct_beta_variance = 1 / n_value
    overcontrolled_beta_variance = diagnostic_sum / (
        n_value * gamma_denominator * gamma_denominator
    )
    collider_coefficient_variance = 1 / (
        n_value * gamma_denominator * gamma_denominator
    )

    overcontrolled_numerator = beta_value - delta_value * gamma_value
    correct_beta_t_statistic = _coefficient_times_square_root(
        "correct beta t-statistic", beta_value, n_value
    )
    overcontrolled_beta_t_statistic = _coefficient_times_square_root(
        "overcontrolled beta t-statistic",
        overcontrolled_numerator,
        n_value / diagnostic_sum,
    )
    collider_t_statistic = _coefficient_times_square_root(
        "collider t-statistic", gamma_value, n_value
    )

    return ColliderDiagnostics(
        residual_variance=_as_result("residual variance", residual_variance),
        outcome_variance=_as_result("outcome variance", outcome_variance),
        correct_r_squared=_as_result("correct R-squared", correct_r_squared),
        overcontrolled_r_squared=_as_result(
            "overcontrolled R-squared", overcontrolled_r_squared
        ),
        correct_adjusted_r_squared=_as_result(
            "correct adjusted R-squared", correct_adjusted_r_squared
        ),
        overcontrolled_adjusted_r_squared=_as_result(
            "overcontrolled adjusted R-squared",
            overcontrolled_adjusted_r_squared,
        ),
        correct_beta_variance=_as_result(
            "correct beta variance", correct_beta_variance
        ),
        overcontrolled_beta_variance=_as_result(
            "overcontrolled beta variance", overcontrolled_beta_variance
        ),
        collider_coefficient_variance=_as_result(
            "collider coefficient variance", collider_coefficient_variance
        ),
        correct_beta_t_statistic=correct_beta_t_statistic,
        overcontrolled_beta_t_statistic=overcontrolled_beta_t_statistic,
        collider_t_statistic=collider_t_statistic,
        adjusted_r_squared_prefers_overcontrolled=(
            gamma_value * gamma_value * (n_integer - 3) > 1
        ),
        absolute_beta_t_prefers_overcontrolled=(
            overcontrolled_numerator * overcontrolled_numerator
            > beta_value * beta_value * diagnostic_sum
        ),
    )
