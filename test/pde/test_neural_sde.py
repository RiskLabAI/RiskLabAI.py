"""Boundary, likelihood and coupled-step checks for the scalar option factor."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr

torch = pytest.importorskip("torch")
method = pytest.importorskip("RiskLabAI.pde.neural_sde")


def _grid():
    strikes, maturities = np.array([0.8, 1, 1.2]), np.array([0.5, 1])

    def prices(volatility):
        scale = volatility * np.sqrt(maturities[:, None])
        d1 = -np.log(strikes)[None, :] / scale + scale / 2
        return (ndtr(d1) - strikes * ndtr(d1 - scale)).ravel()

    offset, loading = prices(0.2), prices(0.3) - prices(0.1)
    matrix, bounds = method.normalized_call_grid_constraints(strikes, maturities)
    interval = method.affine_price_state_interval(offset, loading, matrix, bounds)
    return offset, loading, matrix, bounds, interval


def test_grid_constraints_black_scholes_and_endpoint_states():
    offset, loading, matrix, bounds, (lower, upper) = _grid()
    assert np.all(matrix @ offset >= bounds - 1e-12)
    assert lower < 0 < upper
    states = np.linspace(lower, upper, 51)
    prices = offset + states[:, None] * loading
    assert np.min(prices @ matrix.T - bounds) >= -1e-12
    for state in (lower - 0.01, upper + 0.01):
        assert np.any(matrix @ (offset + state * loading) < bounds - 1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_hard_boundary_conditions(seed):
    model = method.IntervalNeuralSDE(
        -2, 3, drift_bound=0.4, diffusion_scale=0.2, seed=seed
    )
    mu, sigma = model(torch.tensor([-2, 3], dtype=torch.float64))
    assert mu[0] >= 0 and mu[1] <= 0
    assert_allclose(sigma.detach(), 0, atol=0)
    for raw in (-100, 100):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-1].bias[0] = raw
        mu, sigma = model(torch.tensor([-2, 3], dtype=torch.float64))
        assert mu[0] >= 0 and mu[1] <= 0
        assert_allclose(sigma.detach(), 0, atol=0)


def test_neural_likelihood_against_normal_density_and_gradients():
    model = method.IntervalNeuralSDE(
        -1, 1, drift_bound=0.4, diffusion_scale=0.2, seed=43
    )
    states = torch.tensor([-0.3, 0.1, 0.7], dtype=torch.float64)
    next_states = states + torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64)
    mu, sigma = model(states)
    distribution = torch.distributions.Normal(states + mu * 0.01, sigma * np.sqrt(0.01))
    expected = -distribution.log_prob(next_states).mean().item()
    loss = model.negative_log_likelihood(states, next_states, 0.01)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    history = method.fit_interval_neural_sde(
        model, states, next_states, dt=0.01, iterations=15, learning_rate=0.01
    )
    assert history[-1] < history[0]
    with pytest.raises(ValueError):
        model.negative_log_likelihood(torch.tensor([-1.0]), torch.tensor([-0.9]), 0.01)


def test_simulation_grid_and_coupled_step_benchmark():
    offset, loading, matrix, bounds, (lower, upper) = _grid()
    model = method.IntervalNeuralSDE(
        lower, upper, drift_bound=0.1, diffusion_scale=0.2, seed=91
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias[1] = np.log(np.expm1(0.03))
    generator = torch.Generator().manual_seed(92)
    fine_dw = torch.randn(
        2048, 128, generator=generator, dtype=torch.float64
    ) / np.sqrt(128)
    coarse_dw = fine_dw.reshape(2048, 32, 4).sum(dim=2)
    initial = torch.zeros(2048, dtype=torch.float64)
    fine = method.simulate_interval_neural_sde(model, initial, fine_dw, dt=1 / 128)
    coarse = method.simulate_interval_neural_sde(model, initial, coarse_dw, dt=1 / 32)
    for paths in (fine, coarse):
        prices = offset + paths.numpy()[:, :, None] * loading
        assert np.min(prices.reshape(-1, len(offset)) @ matrix.T - bounds) > -1e-12
        final_prices = prices[:, -1]
        standard_error = final_prices.std(axis=0, ddof=1) / np.sqrt(len(paths))
        assert np.all(np.abs(final_prices.mean(axis=0) - offset) <= 5 * standard_error)
    assert torch.sqrt(torch.mean((fine[:, -1] - coarse[:, -1]) ** 2)).item() < 0.001
    with pytest.raises(RuntimeError, match="Euler step"):
        method.simulate_interval_neural_sde(
            model, initial[:1], torch.full((1, 1), 1000.0, dtype=torch.float64), dt=1
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: method.normalized_call_grid_constraints([1, 0.5], [1]),
        lambda: method.normalized_call_grid_constraints([0.5, 1], [0]),
        lambda: method.affine_price_state_interval([1], [0], [[1]], [2]),
        lambda: method.affine_price_state_interval([1], [1], [[1]], [0]),
        lambda: method.IntervalNeuralSDE(1, 1, drift_bound=1, diffusion_scale=1),
    ],
)
def test_invalid_grid_or_state_contract(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("angle", [0, 0.3, 1.1])
def test_rotated_polytope_faces_and_vertices(angle):
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    normals = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]]) @ rotation
    bounds = np.array([-1, -1, -1, -1, -2])
    model = method.PolytopeNeuralSDE(
        normals, bounds, [0, 0], drift_bound=0.4, diffusion_scale=0.2, seed=82
    )
    points = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]) @ rotation
    drift, diffusion = model(torch.tensor(points, dtype=torch.float64))
    for i, point in enumerate(points):
        active = np.isclose(normals @ point, bounds, atol=1e-12)
        assert np.min(normals[active] @ drift[i].detach().numpy()) >= -1e-12
        assert_allclose(normals[active] @ diffusion[i].detach().numpy(), 0, atol=1e-12)


def test_polytope_likelihood_training_and_exit_handling():
    model = method.PolytopeNeuralSDE(
        [[1, 0], [0, 1], [-1, -1]],
        [0, 0, -1],
        [0.3, 0.3],
        drift_bound=0.1,
        diffusion_scale=0.2,
        seed=11,
    )
    states = torch.tensor([[0.2, 0.3], [0.4, 0.2], [0.1, 0.5]], dtype=torch.float64)
    following = states + torch.tensor(
        [[0.01, -0.02], [-0.01, 0.01], [0.01, 0.01]], dtype=torch.float64
    )
    drift, diffusion = model(states)
    distribution = torch.distributions.MultivariateNormal(
        states + 0.01 * drift,
        covariance_matrix=0.01 * diffusion @ diffusion.transpose(-1, -2),
    )
    expected = -distribution.log_prob(following).mean().item()
    assert model.negative_log_likelihood(
        states, following, 0.01
    ).item() == pytest.approx(expected)
    losses = method.fit_polytope_neural_sde(
        model, states, following, dt=0.01, iterations=5
    )
    assert np.isfinite(losses).all()
    increments = torch.zeros(3, 5, 2, dtype=torch.float64)
    paths = method.simulate_polytope_neural_sde(model, states, increments, dt=0.001)
    assert paths.shape == (3, 6, 2)
    with pytest.raises(RuntimeError, match="Euler step"):
        method.simulate_polytope_neural_sde(model, states, increments + 100, dt=0.001)


def test_hjm_residual_detects_incompatible_price_drift():
    basis = np.array([[1, 0], [0, 1], [1, 1]])
    drift, diffusion = np.array([0.2, -0.3]), np.diag([0.1, 0.2])
    risk_price = np.array([0.4, 0.7])
    required = basis @ (drift - diffusion @ risk_price)
    result = method.hjm_drift_residual(basis, drift, diffusion, required)
    assert_allclose(result["market_price_of_risk"], risk_price)
    assert result["squared_residual"] < 1e-25
    required[-1] += 1
    assert (
        method.hjm_drift_residual(basis, drift, diffusion, required)["squared_residual"]
        > 0.1
    )


def _constant_diffusion(model, matrix):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        rows, columns = np.tril_indices(model.dimension)
        values = matrix[rows, columns].copy()
        diagonal = rows == columns
        values[diagonal] = np.log(np.expm1(values[diagonal]))
        model.network[-1].bias[model.dimension :] = torch.tensor(values)


def _projector_integral(normals, distances, scale):
    """Independent SVD evaluation of the subspace-projector integral."""
    dimension = normals.shape[1]
    result, previous = np.zeros((dimension, dimension)), 0.0
    identity = np.eye(dimension)
    for level in np.unique(distances):
        earlier = normals[distances < level]
        projector = np.zeros_like(identity)
        if len(earlier):
            _, singular, vectors = np.linalg.svd(earlier, full_matrices=False)
            basis = vectors[singular > 1e-10]
            projector = basis.T @ basis
        value = level / (scale + level)
        result += (value - previous) * (identity - projector)
        previous = value
    return result


def test_anisotropic_covariance_is_continuous_at_nearest_face_tie():
    model = method.PolytopeNeuralSDE(
        [[1, 0], [-1, 0], [0, 1], [0, -1]],
        [-1] * 4,
        [0, 0],
        drift_bound=0.1,
        diffusion_scale=0.2,
    )
    _constant_diffusion(model, np.diag([0.1, 0.2]))
    for epsilon in (1e-3, 1e-6, 1e-9):
        states = torch.tensor(
            [[0.2 + epsilon, 0.2], [0.2, 0.2 + epsilon]], dtype=torch.float64
        )
        _, diffusion = model(states)
        covariance = (diffusion @ diffusion.transpose(-1, -2)).detach().numpy()
        g = (0.8 - epsilon) / (1 - epsilon)
        expected = [
            np.diag([0.01 * g**2, 0.04 * 0.8**2]),
            np.diag([0.01 * 0.8**2, 0.04 * g**2]),
        ]
        assert_allclose(covariance, expected, rtol=1e-12, atol=1e-15)
        assert np.linalg.norm(covariance[0] - covariance[1]) < 0.02 * epsilon


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_oblique_shrinking_matches_projector_integral_and_row_permutations(dimension):
    rng = np.random.default_rng(761 + dimension)
    normals = np.r_[
        np.eye(dimension), -np.eye(dimension), rng.normal(size=(5, dimension))
    ]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.r_[normals, normals[[0]]]
    bounds = np.full(len(normals), -1.0)
    model = method.PolytopeNeuralSDE(
        normals, bounds, np.zeros(dimension), drift_bound=0.1, diffusion_scale=0.3
    )
    base = np.tril(rng.normal(size=(dimension, dimension)) * 0.05)
    np.fill_diagonal(base, np.arange(1, dimension + 1) * 0.1)
    _constant_diffusion(model, base)
    points = np.r_[np.zeros((1, dimension)), rng.normal(size=(12, dimension)) * 0.05]
    states = torch.tensor(points, dtype=torch.float64)
    _, observed = model(states)
    expected = np.array(
        [
            _projector_integral(normals, normals @ point - bounds, 0.3) @ base
            for point in points
        ]
    )
    assert_allclose(observed.detach(), expected, atol=1e-13)
    permutation = rng.permutation(len(normals))
    reordered = method.PolytopeNeuralSDE(
        normals[permutation],
        bounds[permutation],
        np.zeros(dimension),
        drift_bound=0.1,
        diffusion_scale=0.3,
    )
    reordered.network.load_state_dict(model.network.state_dict())
    mu1, sigma1 = model(states)
    mu2, sigma2 = reordered(states)
    assert_allclose(mu1.detach(), mu2.detach(), atol=1e-14)
    assert_allclose(sigma1.detach(), sigma2.detach(), atol=1e-13)
    for epsilon in (1e-4, 1e-7, 1e-10):
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        _, paired = model(
            torch.tensor(
                np.array([epsilon * direction, -epsilon * direction]),
                dtype=torch.float64,
            )
        )
        assert torch.linalg.matrix_norm(paired[0] - paired[1]) < 20 * epsilon


def test_nonsimple_vertex_and_linear_boundary_approach():
    normals = np.array(
        [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1], [0, 0, -1]], dtype=float
    )
    bounds = np.array([0, 0, 0, 0, -1], dtype=float)
    model = method.PolytopeNeuralSDE(
        normals, bounds, [0, 0, 0.5], drift_bound=0.1, diffusion_scale=0.2
    )
    states = torch.tensor(
        [[0, 0, 0], [0.3, 0, 0.3], [0, 0, 1]], dtype=torch.float64, requires_grad=True
    )
    drift, diffusion = model(states)
    for index, state in enumerate(states.detach().numpy()):
        active = np.isclose(normals @ state - bounds, 0)
        assert_allclose(
            normals[active] @ diffusion[index].detach().numpy(), 0, atol=1e-13
        )
        assert np.min(normals[active] @ drift[index].detach().numpy()) >= -1e-12
    diffusion.square().sum().backward()
    assert torch.isfinite(states.grad).all()
    for epsilon in (1e-4, 1e-7, 1e-10, 1e-13):
        _, sigma = model(torch.tensor([[0, 0, epsilon]], dtype=torch.float64))
        ratio = torch.linalg.matrix_norm(sigma[0]).item() / epsilon
        assert 0 < ratio < 100


def test_scalar_linear_boundary_and_polytope_agreement():
    scalar = method.IntervalNeuralSDE(-1, 1, drift_bound=0.1, diffusion_scale=0.2)
    polytope = method.PolytopeNeuralSDE(
        [[1], [-1]], [-1, -1], [0], drift_bound=0.1, diffusion_scale=0.2
    )
    _constant_diffusion(polytope, np.array([[0.2]]))
    with torch.no_grad():
        for parameter in scalar.parameters():
            parameter.zero_()
        scalar.network[-1].bias[1] = np.log(np.expm1(0.2))
    states = torch.tensor(
        [-1, -1 + 1e-8, -0.2, 0.0, 0.4, 1], dtype=torch.float64, requires_grad=True
    )
    mu, sigma = scalar(states)
    other_mu, other_sigma = polytope(states[:, None])
    assert_allclose(mu.detach(), other_mu[:, 0].detach())
    assert_allclose(sigma.detach(), other_sigma[:, 0, 0].detach())
    sigma.sum().backward()
    assert torch.isfinite(states.grad).all()
    assert abs(states.grad[1].item()) <= 1.01


def test_call_grid_witness_reprices_and_has_ordered_marginals():
    offset, _, _, _, _ = _grid()
    strikes, maturities = np.array([0.8, 1, 1.2]), np.array([0.5, 1])
    result = method.normalized_call_grid_witness(
        strikes, maturities, offset.reshape(2, 3)
    )
    support, probabilities = result["support"], result["probabilities"]
    assert np.all(probabilities >= 0)
    assert_allclose(probabilities.sum(axis=1), 1)
    assert_allclose(probabilities @ support, 1)
    calls = probabilities @ np.maximum(support[:, None] - strikes, 0)
    assert_allclose(calls, offset.reshape(2, 3), atol=1e-12)
    dense_strikes = np.linspace(0, support[-1] * 1.1, 301)
    dense_calls = probabilities @ np.maximum(support[:, None] - dense_strikes, 0)
    assert np.min(np.diff(dense_calls, axis=0)) >= -1e-12
    deterministic = method.normalized_call_grid_witness([0.5, 1, 2], [1], [[0.5, 0, 0]])
    assert_allclose(deterministic["probabilities"] @ deterministic["support"], 1)
    with pytest.raises(ValueError, match="positive flat tail"):
        method.normalized_call_grid_witness([1, 2], [1], [[0.1, 0.1]])
    with pytest.raises(ValueError, match="inequalities"):
        method.normalized_call_grid_witness([1, 2], [1], [[0.1, 0.2]])
    with pytest.raises(ValueError, match="shape"):
        method.normalized_call_grid_witness([1, 2], [1], [[0.1, 0.05], [0.1, 0.05]])


def test_multifactor_grid_martingale_levels_and_coupled_steps():
    """Check a bounded finite-grid martingale, not a dynamic payoff-pricing claim."""
    offset, loading, matrix, bounds, _ = _grid()
    basis = np.column_stack([loading, loading * np.repeat([-1, 1], 3)])
    normals, state_bounds = matrix @ basis, bounds - matrix @ offset
    nonzero = np.linalg.norm(normals, axis=1) > 1e-12
    model = method.PolytopeNeuralSDE(
        normals[nonzero],
        state_bounds[nonzero],
        [0, 0],
        drift_bound=0.1,
        diffusion_scale=0.2,
    )
    _constant_diffusion(model, np.diag([0.02, 0.015]))
    increments = (
        torch.randn(
            128,
            64,
            2,
            generator=torch.Generator().manual_seed(892),
            dtype=torch.float64,
        )
        / 8
    )
    initial = torch.zeros(128, 2, dtype=torch.float64)
    fine = method.simulate_polytope_neural_sde(model, initial, increments, dt=1 / 64)
    coarse = method.simulate_polytope_neural_sde(
        model, initial, increments.reshape(128, 16, 4, 2).sum(dim=2), dt=1 / 16
    )
    assert torch.sqrt(torch.mean((fine[:, -1] - coarse[:, -1]) ** 2)) < 0.001
    for paths in (fine, coarse):
        prices = offset + paths.numpy() @ basis.T
        assert np.min(prices.reshape(-1, len(offset)) @ matrix.T - bounds) > 0
        terminal = prices[:, -1]
        errors = terminal.mean(axis=0) - offset
        standard_errors = terminal.std(axis=0, ddof=1) / np.sqrt(len(terminal))
        assert np.all(np.abs(errors) < 5 * standard_errors)
    for prices in offset + fine[:, -1].numpy() @ basis.T:
        witness = method.normalized_call_grid_witness(
            [0.8, 1, 1.2], [0.5, 1], prices.reshape(2, 3)
        )
        assert witness["maximum_price_error"] < 1e-12
