"""Independent mathematical checks for the bounded book-method contracts."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from RiskLabAI.features.entropy_features.ambiguity import (
    probability_ambiguity,
    normal_prior_bin_probabilities,
)
from RiskLabAI.features.microstructural_features.order_flow import (
    hawkes_intensity,
    hawkes_integrated_intensity,
    hawkes_log_likelihood,
    simulate_hawkes,
    fit_hawkes,
    volume_synchronized_pin,
)
from RiskLabAI.backtest.tail_risk import (
    generalized_pareto_tail_risk,
    fit_pot_tail_risk,
    quantile_exception_tests,
)
from RiskLabAI.optimization.constrained import (
    constrained_minimum_variance,
    robust_mean_variance,
)
from RiskLabAI.optimization.decision_focused import spo_plus_loss, spo_plus_torch
from RiskLabAI.causal_factor_analysis.bounds_transport import (
    manski_ate_bounds,
    transported_ate,
)
from RiskLabAI.causal_factor_analysis.interference import exposure_mean_contrast
from RiskLabAI.causal_factor_analysis.policy_evaluation import (
    evaluate_policy,
    select_policy,
)


def test_probability_ambiguity_enumeration():
    p = np.array([[0.2, 0.8], [0.6, 0.4], [1, 0]])
    w = np.array([0.2, 0.3, 0.5])
    expected = sum(
        sum(w[i] * p[i, k] for i in range(3))
        * sum(
            w[i] * w[j] * (p[i, k] - p[j, k]) ** 2 / 2
            for i in range(3)
            for j in range(3)
        )
        for k in range(2)
    )
    result = probability_ambiguity(p, prior_weights=w, bin_width=0.002)
    assert result["unscaled"] == pytest.approx(expected)
    assert result["scaled"] == pytest.approx(expected / (0.002 * 0.998))
    assert probability_ambiguity(p[::-1], prior_weights=w[::-1])[
        "unscaled"
    ] == pytest.approx(expected)
    assert probability_ambiguity([[0.5, 0.5]] * 3)["unscaled"] == 0
    assert probability_ambiguity([[0, 1], [1, 0]])["unscaled"] == 0.25


def test_normal_bins_and_units():
    p = normal_prior_bin_probabilities([0, 0.01, -100], [0.01, 0.02, 0.1])
    assert p.shape == (3, 62)
    assert_allclose(p.sum(axis=1), 1)
    assert_allclose(p[0], p[0, ::-1], atol=1e-15)
    assert np.all(p >= 0)
    assert_allclose(
        normal_prior_bin_probabilities([1], [2], edges=[-3, 0, 4]),
        normal_prior_bin_probabilities([100], [200], edges=[-300, 0, 400]),
    )


@pytest.mark.parametrize("mu,alpha,beta", [(0.8, 0, 1), (0.3, 0.7, 2), (3, 0.01, 0.02)])
def test_hawkes_integral_and_direct_likelihood(mu, alpha, beta):
    events = np.array([0.2, 0.7, 1.4, 2.0])
    horizon = 3.0
    integral = quad(
        lambda t: mu
        + sum(alpha * np.exp(-beta * (t - event)) for event in events if event < t),
        0,
        horizon,
        points=events,
    )[0]
    assert hawkes_integrated_intensity(
        events, horizon, mu, alpha, beta
    ) == pytest.approx(integral)
    direct = (
        sum(
            np.log(
                mu
                + sum(
                    alpha * np.exp(-beta * (event - past))
                    for past in events
                    if past < event
                )
            )
            for event in events
        )
        - integral
    )
    assert hawkes_log_likelihood(events, horizon, mu, alpha, beta) == pytest.approx(
        direct
    )
    assert_allclose(hawkes_intensity([0, 0.2], events, mu, alpha, beta), mu)


def test_hawkes_simulation_and_fit():
    events = simulate_hawkes(100, 1.0, 0.4, 1.2, random_state=93)
    assert_allclose(events, simulate_hawkes(100, 1.0, 0.4, 1.2, random_state=93))
    assert np.all(np.diff(events) > 0) and np.all((events >= 0) & (events <= 100))
    fit = fit_hawkes(events, 100)
    assert 0 <= fit["alpha"] < fit["beta"] and fit["mu"] > 0
    poisson = len(events) * np.log(len(events) / 100) - len(events)
    assert fit["log_likelihood"] >= poisson - 1e-5
    assert hawkes_log_likelihood([], 3, 2, 0, 1) == -6
    with pytest.raises(ValueError):
        hawkes_log_likelihood([1, 1], 3, 2, 0, 1)


def test_vpin_completed_buckets():
    assert_allclose(
        volume_synchronized_pin(
            [5, 10, 0, 5], [5, 0, 10, 5], bucket_volume=10, window=2
        ),
        [0.5, 1, 0.5],
    )
    with pytest.raises(ValueError):
        volume_synchronized_pin([5, 8], [5, 0], bucket_volume=10, window=2)


@pytest.mark.parametrize("shape", [-0.4, 0, 0.3, 0.9])
def test_pareto_tail_integral(shape):
    from scipy.stats import genpareto

    result = generalized_pareto_tail_risk(2, 0.1, shape, 0.7, 0.98)
    q = result["value_at_risk"] - 2
    upper = np.inf if shape >= 0 else -0.7 / shape
    excess_mean = quad(
        lambda x: genpareto.sf(x, shape, scale=0.7) / genpareto.sf(q, shape, scale=0.7),
        q,
        upper,
    )[0]
    assert result["expected_shortfall"] == pytest.approx(
        result["value_at_risk"] + excess_mean, rel=1e-6
    )
    assert genpareto.cdf(q, shape, scale=0.7) == pytest.approx(0.8)


def test_pareto_fit_and_domains():
    from scipy.stats import genpareto

    losses = np.r_[
        np.zeros(100), 2 + genpareto.ppf(np.linspace(0.001, 0.999, 500), 0, scale=0.7)
    ]
    result = fit_pot_tail_risk(losses, threshold=2, probability=0.98)
    assert abs(result["shape"]) < 0.03
    assert result["scale"] == pytest.approx(0.7, rel=0.03)
    assert np.isinf(
        generalized_pareto_tail_risk(0, 0.1, 1, 1, 0.99)["expected_shortfall"]
    )
    with pytest.raises(ValueError):
        generalized_pareto_tail_risk(0, 0.1, 0, 1, 0.5)


def test_exception_likelihood_degenerate_and_alternating():
    result = quantile_exception_tests([0, 1, 0, 1, 0], exception_probability=0.4)
    assert result["coverage_statistic"] == pytest.approx(0)
    assert_allclose(result["transitions"], [[0, 2], [2, 0]])
    assert result["independence_statistic"] == pytest.approx(8 * np.log(2))
    for value in (0, 1):
        result = quantile_exception_tests([value] * 4, exception_probability=0.2)
        assert result["independence_statistic"] is None
        assert result["coverage_statistic"] == pytest.approx(
            -8 * np.log(0.2 if value else 0.8)
        )


@pytest.mark.parametrize("scale", [1e-8, 1, 1e8])
def test_constrained_portfolio_analytic(scale):
    pytest.importorskip("cvxpy")
    result = constrained_minimum_variance(np.diag([1, 2, 4]) * scale)
    assert_allclose(result["weights"], [4 / 7, 2 / 7, 1 / 7], atol=1e-5)
    constrained = constrained_minimum_variance(
        np.eye(2) * scale, expected_returns=[0, 1], return_floor=0.8
    )
    assert_allclose(constrained["weights"], [0.2, 0.8], atol=1e-5)
    with pytest.raises(ValueError):
        constrained_minimum_variance(
            np.eye(2), expected_returns=[0, 1], return_floor=0.8, caps=[1, 0.5]
        )
    with pytest.raises(ValueError):
        constrained_minimum_variance([[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        constrained_minimum_variance(np.diag([-1, 2]) * scale * 1e-15)
    with pytest.raises(ValueError):
        constrained_minimum_variance(np.array([[1, 1], [0, 1]]) * scale * 1e-15)


def test_robust_portfolio_grid_oracle():
    pytest.importorskip("cvxpy")
    mean, covariance, loading = (
        np.array([0.1, 0.2]),
        np.diag([0.2, 0.3]),
        np.diag([0.03, 0.08]),
    )
    for radius in (0, 1.2):
        result = robust_mean_variance(
            mean, covariance, loading, radius=radius, risk_aversion=2
        )
        grid = np.column_stack((np.linspace(0, 1, 10001), np.linspace(1, 0, 10001)))
        objective = (
            grid @ mean
            - radius * np.linalg.norm(grid @ loading, axis=1)
            - np.einsum("ni,ij,nj->n", grid, covariance, grid)
        )
        assert result["objective"] >= objective.max() - 1e-8
        w = result["weights"]
        direction = -loading.T @ w / np.linalg.norm(loading.T @ w)
        assert w @ (mean + radius * loading @ direction) == pytest.approx(
            result["worst_case_expected_return"]
        )


def test_spo_subgradient_and_torch():
    torch = pytest.importorskip("torch")
    vertices = np.array([[0, 0], [1, 0], [0, 1], [0.2, 0.8]])
    actual, predicted = np.array([2, -1]), np.array([-0.4, 0.8])
    result = spo_plus_loss(predicted, actual, vertices)
    for shifted in np.random.default_rng(54).normal(size=(30, 2)):
        assert (
            spo_plus_loss(shifted, actual, vertices)["loss"]
            >= result["loss"] + result["subgradient"] @ (shifted - predicted) - 1e-12
        )
    prediction = torch.tensor(predicted, requires_grad=True)
    loss = spo_plus_torch(
        prediction, torch.tensor(actual, dtype=torch.float64), torch.tensor(vertices)
    )
    loss.backward()
    assert loss.item() == pytest.approx(result["loss"])
    assert_allclose(prediction.grad, result["subgradient"])
    assert spo_plus_loss(actual, actual, vertices)["loss"] == pytest.approx(0)


def test_manski_extreme_counterfactual_enumeration():
    observed, treatment = np.array([0.2, 0.8, 0.5]), np.array([1, 0, 1])
    effects = []
    for missing in itertools.product([0, 1], repeat=3):
        y1 = np.where(treatment, observed, missing)
        y0 = np.where(treatment, missing, observed)
        effects.append(np.mean(y1 - y0))
    bounds = manski_ate_bounds(observed, treatment, lower=0, upper=1)
    assert bounds["lower"] == min(effects)
    assert bounds["upper"] == max(effects)


def test_transport_standardization_support():
    assert (
        transported_ate([2, 4, 8], [1, 2, 9], [0.25, 0.75, 0], [2, 4, 0], [3, 7, 0])
        == 1.75
    )
    with pytest.raises(ValueError):
        transported_ate([2, 4], [1, 2], [0.25, 0.75], [2, 0], [3, 7])


def test_exposure_contrast_exact_randomized_design():
    assignments = list(itertools.product([0, 1], repeat=3))
    exposure = np.array(
        [[int(a[(i + 1) % 3] or a[(i + 2) % 3]) for i in range(3)] for a in assignments]
    )
    probabilities = np.array([[0.25, 0.75]] * 3)
    potential = np.array([[1, 2], [3, 7], [2, 1]])
    estimates = []
    for realized in exposure:
        y = potential[np.arange(3), realized]
        estimates.append(
            exposure_mean_contrast(y, realized, probabilities, first=1, second=0)[
                "contrast"
            ]
        )
    assert np.mean(estimates) == pytest.approx(
        np.mean(potential[:, 1] - potential[:, 0])
    )


def test_policy_double_robustness_exact_population():
    actions = np.array([0, 0, 0, 1])
    rewards = np.array([2, 2, 2, 6])
    true_behavior = np.array([[0.75, 0.25]] * 4)
    policy = np.array([[0.4, 0.6]] * 4)
    correct_q = np.array([[2, 6]] * 4)
    for behavior, q in (
        (true_behavior, np.zeros((4, 2))),
        (np.ones((4, 2)) / 2, correct_q),
    ):
        result = evaluate_policy(actions, rewards, behavior, policy, q)
        assert result["dr_value"] == pytest.approx(4.4)
    with pytest.raises(ValueError):
        evaluate_policy([0], [1], [[1, 0]], [[0, 1]], [[1, 1]])


def test_policy_selection_does_not_use_evaluation_outcomes():
    actions = np.array([0, 1, 0, 1])
    behavior = np.ones((4, 2)) / 2
    policies = np.stack([np.tile([1, 0], (4, 1)), np.tile([0, 1], (4, 1))])
    for rewards in ([1, 3, 100, -100], [1, 3, -100, 100]):
        result = select_policy(
            actions,
            rewards,
            behavior,
            policies,
            np.zeros((4, 2)),
            training_indices=[0, 1],
            evaluation_indices=[2, 3],
        )
        assert result["selected_policy"] == 1
    with pytest.raises(ValueError):
        select_policy(
            actions,
            rewards,
            behavior,
            policies,
            np.zeros((4, 2)),
            training_indices=[0, 2],
            evaluation_indices=[1, 3],
        )


def test_sensitivity_supplier_against_omitted_variable_ols():
    sensemakr = pytest.importorskip("sensemakr")
    import statsmodels.api as sm

    rng = np.random.default_rng(65)
    x, z, noise, noise_y = rng.normal(size=(4, 500))
    d = 0.8 * z + 0.3 * x + noise
    y = 2 * d + 1.4 * z + x + noise_y
    reduced = sm.OLS(y, sm.add_constant(np.column_stack([d, x]))).fit()
    full = sm.OLS(y, sm.add_constant(np.column_stack([d, x, z]))).fit()
    d_reduced = sm.OLS(d, sm.add_constant(x)).fit()
    d_full = sm.OLS(d, sm.add_constant(np.column_stack([x, z]))).fit()
    r2d = 1 - d_full.ssr / d_reduced.ssr
    r2y = 1 - full.ssr / reduced.ssr
    adjusted = sensemakr.adjusted_estimate(
        float(r2d),
        float(r2y),
        estimate=float(reduced.params[1]),
        se=float(reduced.bse[1]),
        dof=int(reduced.df_resid),
    )
    adjusted_se = sensemakr.adjusted_se(
        float(r2d), float(r2y), se=float(reduced.bse[1]), dof=int(reduced.df_resid)
    )
    assert float(np.asarray(adjusted).item()) == pytest.approx(full.params[1])
    assert float(np.asarray(adjusted_se).item()) == pytest.approx(full.bse[1])
    assert (
        float(
            np.asarray(
                sensemakr.adjusted_estimate(0, 0, estimate=2, se=0.1, dof=30)
            ).item()
        )
        == 2
    )


def test_markov_supplier_filter_has_no_future_information():
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    data = np.random.default_rng(4).normal(size=50)
    parameters = np.array([0.9, 0.8, -1, 1, 0.5, 2])
    full = (
        MarkovRegression(data, k_regimes=2, trend="c", switching_variance=True)
        .filter(parameters)
        .filtered_marginal_probabilities
    )
    prefix = (
        MarkovRegression(data[:25], k_regimes=2, trend="c", switching_variance=True)
        .filter(parameters)
        .filtered_marginal_probabilities
    )
    assert_allclose(full[:25], prefix)
    assert_allclose(full.sum(axis=1), 1)
    swapped = np.array(
        [
            1 - parameters[1],
            1 - parameters[0],
            parameters[3],
            parameters[2],
            parameters[5],
            parameters[4],
        ]
    )
    reversed_states = (
        MarkovRegression(data, k_regimes=2, trend="c", switching_variance=True)
        .filter(swapped)
        .filtered_marginal_probabilities
    )
    assert_allclose(full, reversed_states[:, ::-1], atol=1e-12)


def test_adaptation_prediction_precedes_current_outcome():
    pytest.importorskip("river")
    from sklearn.dummy import DummyRegressor
    from RiskLabAI.backtest.adaptation import prequential_adaptation

    x = np.arange(250)[:, None]
    y = np.r_[np.zeros(100), np.ones(150)]
    arguments = dict(
        estimator=DummyRegressor(),
        loss_function=lambda actual, prediction: abs(actual - prediction),
        window_size=30,
        min_samples=20,
        delta=0.1,
    )
    full = prequential_adaptation(x, y, **arguments)
    assert len(full["alarm_indices"]) > 0
    for endpoint in (101, 160):
        prefix = prequential_adaptation(x[:endpoint], y[:endpoint], **arguments)
        assert_allclose(
            full["predictions"][:endpoint], prefix["predictions"], equal_nan=True
        )
    changed = y.copy()
    changed[100] = 0
    altered = prequential_adaptation(x, changed, **arguments)
    assert full["predictions"][100] == altered["predictions"][100]
