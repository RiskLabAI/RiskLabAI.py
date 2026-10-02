"""Cash-flow, information-boundary and objective checks for learned methods."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

torch = pytest.importorskip("torch")

hedging = pytest.importorskip("RiskLabAI.backtest.deep_hedging")
generation = pytest.importorskip("RiskLabAI.data.synthetic_data.timegan")
gbm_hedging_paths = hedging.gbm_hedging_paths
hedging_pnl = hedging.hedging_pnl
entropic_risk = hedging.entropic_risk
black_scholes_call_delta = hedging.black_scholes_call_delta
DeepHedge = hedging.DeepHedge
train_deep_hedge = hedging.train_deep_hedge
chronological_timegan_data = generation.chronological_timegan_data
TimeGAN = generation.TimeGAN
fit_timegan = generation.fit_timegan
timegan_quality_report = generation.timegan_quality_report


def test_cash_flow_from_cash_account_and_terminal_sale():
    prices = torch.tensor([[100, 110, 105], [100, 90, 95]], dtype=torch.float64)
    positions = torch.tensor(
        [[0.4, 0.8], [-0.2, 0.1]], dtype=torch.float64, requires_grad=True
    )
    result = hedging_pnl(prices, positions, strike=100, cost_rate=0.01, premium=3)
    expected = []
    for row in range(2):
        cash, holding = 3, 0
        for index in range(2):
            desired = positions.detach().numpy()[row, index]
            trade = desired - holding
            cash -= (
                trade * prices[row, index].item()
                + abs(trade) * prices[row, index].item() * 0.01
            )
            holding = desired
        cash += (
            holding * prices[row, -1].item()
            - abs(holding) * prices[row, -1].item() * 0.01
        )
        cash -= max(prices[row, -1].item() - 100, 0)
        expected.append(cash)
    assert_allclose(result.detach(), expected, atol=1e-13)
    result.sum().backward()
    assert torch.isfinite(positions.grad).all()
    zero = hedging_pnl(prices, torch.zeros_like(positions), strike=100, cost_rate=0.5)
    assert_allclose(zero, [-5, 0])


def test_entropic_risk_translation_and_independent_exponential():
    pnl = torch.tensor([-2, 1, 3], dtype=torch.float64)
    expected = np.log(np.exp(-0.7 * pnl.numpy()).mean()) / 0.7
    assert entropic_risk(pnl, 0.7).item() == pytest.approx(expected)
    assert entropic_risk(pnl + 50000, 0.7).item() == pytest.approx(expected - 50000)
    assert entropic_risk(pnl, 1e-6).item() == pytest.approx(
        -pnl.mean().item(), abs=1e-5
    )


def test_gbm_exact_deterministic_and_local_seed():
    before = torch.random.get_rng_state().clone()
    arguments = dict(
        spot=100, maturity=1, volatility=0, drift=0.1, steps=4, paths=3, seed=24
    )
    paths = gbm_hedging_paths(**arguments)
    assert_allclose(paths, np.tile(100 * np.exp(np.arange(5) * 0.025), (3, 1)))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(paths, gbm_hedging_paths(**arguments))


def test_hedge_causal_features_and_seed():
    before = torch.random.get_rng_state().clone()
    model = DeepHedge(5, seed=64)
    assert torch.equal(before, torch.random.get_rng_state())
    prices = gbm_hedging_paths(
        spot=1, maturity=1, volatility=0.2, drift=0, steps=8, paths=5, seed=13
    )
    changed = prices.clone()
    changed[:, 4:] *= 2
    positions = model(prices, strike=1, maturity=1)
    altered = model(changed, strike=1, maturity=1)
    assert_allclose(positions[:, :4].detach(), altered[:, :4].detach(), atol=0)
    twin = DeepHedge(5, seed=64)
    assert_allclose(
        positions.detach(), twin(prices, strike=1, maturity=1).detach(), atol=0
    )


def test_delta_hedging_refinement_against_call_price():
    from scipy.special import ndtr

    fine = gbm_hedging_paths(
        spot=1, maturity=1, volatility=0.2, drift=0, steps=128, paths=2048, seed=821
    )
    option_price = ndtr(0.1) - ndtr(-0.1)
    errors = []
    for stride in (16, 1):
        prices = fine[:, ::stride]
        positions = black_scholes_call_delta(
            prices, strike=1, maturity=1, volatility=0.2
        )
        pnl = hedging_pnl(
            prices, positions, strike=1, cost_rate=0, premium=option_price
        )
        errors.append(pnl.square().mean().item())
        assert abs(pnl.mean().item()) < 0.004
    assert errors[1] < errors[0] / 4


def test_deep_hedge_training_and_independent_evaluation():
    train = gbm_hedging_paths(
        spot=1, maturity=1, volatility=0.2, drift=0, steps=8, paths=128, seed=61
    )
    evaluation = gbm_hedging_paths(
        spot=1, maturity=1, volatility=0.2, drift=0, steps=8, paths=256, seed=62
    )
    model = DeepHedge(8, seed=63)
    history = train_deep_hedge(
        model,
        train,
        strike=1,
        maturity=1,
        cost_rate=0.002,
        risk_aversion=3,
        iterations=20,
        learning_rate=0.01,
    )
    assert np.isfinite(history).all()
    assert history[-1] < history[0]
    for positions in (
        model(evaluation, strike=1, maturity=1),
        black_scholes_call_delta(evaluation, strike=1, maturity=1, volatility=0.2),
    ):
        risk = entropic_risk(
            hedging_pnl(evaluation, positions, strike=1, cost_rate=0.002), 3
        )
        assert torch.isfinite(risk)


def test_timegan_scaling_and_nonoverlapping_split():
    series = np.column_stack([np.arange(20), np.ones(20)])
    result = chronological_timegan_data(series, training_end=12, window_length=4)
    assert_allclose(result["minimum"], [0, 1])
    assert_allclose(result["scale"], [11, 1])
    assert_allclose(
        result["training"][-1] * result["scale"] + result["minimum"], series[8:12]
    )
    assert_allclose(
        result["validation"][0] * result["scale"] + result["minimum"], series[12:16]
    )
    changed = series.copy()
    changed[12:] *= 100
    altered = chronological_timegan_data(changed, training_end=12, window_length=4)
    assert_allclose(result["training"], altered["training"])
    assert_allclose(result["scale"], altered["scale"])


def test_timegan_objectives_independent_arrays_and_gradients():
    model = TimeGAN(2, 4, seed=1)
    rng = np.random.default_rng(42)
    observed = torch.tensor(rng.uniform(size=(3, 5, 2)), dtype=torch.float64)
    noise = torch.tensor(rng.normal(size=(3, 5, 4)), dtype=torch.float64)
    losses = model.losses(observed, noise)
    with torch.no_grad():
        h = model.embedding(observed)
        reconstructed = model.recovery(h).numpy()
        synthetic = model.latent_sequence(noise)
        teacher = model.latent_sequence(noise, h).numpy()
        real_logits = model.discriminator(h).numpy()
        fake_logits = model.discriminator(synthetic).numpy()
    reconstruction = (
        sum(np.sum((a - b) ** 2) for a, b in zip(observed.numpy(), reconstructed)) / 3
    )
    supervised = sum(np.sum((a - b) ** 2) for a, b in zip(h.numpy(), teacher)) / 3
    discriminator = (
        np.logaddexp(0, -real_logits).sum() / 3 + np.logaddexp(0, fake_logits).sum() / 3
    )
    adversarial = -np.logaddexp(0, fake_logits).sum() / 3
    for key, value in zip(
        ["reconstruction", "supervised", "discriminator", "generator_adversarial"],
        [reconstruction, supervised, discriminator, adversarial],
    ):
        assert losses[key].item() == pytest.approx(value)
    sum(losses.values()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    changed = noise.clone()
    changed[:, 3:] += 100
    assert_allclose(
        model.latent_sequence(noise)[:, :3].detach(),
        model.latent_sequence(changed)[:, :3].detach(),
        atol=0,
    )


def test_timegan_three_phase_determinism_and_quality_report():
    before = torch.random.get_rng_state().clone()
    model = TimeGAN(1, 4, seed=8)
    assert torch.equal(before, torch.random.get_rng_state())
    series = np.sin(np.arange(100) / 4)[:, None]
    data = chronological_timegan_data(series, training_end=60, window_length=6)
    settings = dict(
        embedding_steps=2, supervised_steps=2, joint_steps=2, batch_size=5, seed=12
    )
    history = fit_timegan(model, data["training"], **settings)
    twin = TimeGAN(1, 4, seed=8)
    assert fit_timegan(twin, data["training"], **settings) == history
    assert [entry["phase"] for entry in history] == ["embedding"] * 2 + [
        "supervised"
    ] * 2 + ["joint"] * 2
    assert_allclose(model.sample(20, 6, seed=10), twin.sample(20, 6, seed=10), atol=0)
    quality = timegan_quality_report(
        data["training"],
        data["validation"],
        model.sample(40, 6, seed=20).numpy(),
        model.sample(40, 6, seed=21).numpy(),
    )
    assert 0 <= quality["balanced_discriminator_accuracy"] <= 1
    assert quality["synthetic_trained_prediction_mse"] >= 0
    assert np.isfinite(quality["lag_one_absolute_gap"]).all()


def test_quality_detects_destroyed_temporal_information():
    rng = np.random.default_rng(891)
    sequences = np.repeat(rng.normal(size=(100, 1, 1)), 5, axis=1)
    independent = rng.normal(size=(100, 5, 1))
    result = timegan_quality_report(
        sequences[:60], sequences[60:], independent[:60], independent[60:]
    )
    assert result["lag_one_absolute_gap"][0] > 0.7
    assert (
        result["synthetic_trained_prediction_mse"]
        > result["real_trained_prediction_mse"] + 0.1
    )
