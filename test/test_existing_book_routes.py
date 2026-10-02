"""Direct mathematical checks for previously indirectly tested book routes."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import leaves_list

from RiskLabAI.data.synthetic_data.simulation import form_true_matrix, simulates_cov_mu
from RiskLabAI.optimization.hrp import (
    cluster_variance,
    quasi_diagonal,
    recursive_bisection,
)


def test_covariance_simulation_preserves_block_spectrum():
    state = np.random.get_state()
    try:
        np.random.seed(740)
        mean, covariance = form_true_matrix(2, 3, 0.4)
        assert mean.shape == (6, 1)
        standard_deviation = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(standard_deviation, standard_deviation)
        assert_allclose(np.linalg.eigvalsh(correlation), [0.6] * 4 + [1.8] * 2)
        simulated_mean, simulated_covariance = simulates_cov_mu(
            np.zeros((2, 1)), np.diag([1, 4]), 50000
        )
        assert_allclose(simulated_mean, 0, atol=0.03)
        assert_allclose(simulated_covariance, np.diag([1, 4]), atol=0.06)
    finally:
        np.random.set_state(state)


def test_hrp_helpers_against_diagonal_allocation_and_scipy_tree():
    covariance = pd.DataFrame(
        np.diag([1, 2, 4, 8]), index=list("abcd"), columns=list("abcd")
    )
    assert cluster_variance(covariance, ["a", "c"]) == pytest.approx(0.8)
    allocation = recursive_bisection(covariance, list("abcd"))
    assert_allclose(allocation, np.array([8, 4, 2, 1]) / 15)
    linkage = np.array([[0, 2, 0.1, 2], [1, 3, 0.2, 2], [4, 5, 0.8, 4]])
    assert quasi_diagonal(linkage) == leaves_list(linkage).tolist()


def test_barenblatt_exact_solution_satisfies_public_driver():
    torch = pytest.importorskip("torch")
    equation = pytest.importorskip("RiskLabAI.pde.equation")
    pde = equation.BlackScholesBarenblatt(
        {"dim": 2, "total_time": 1, "num_time_interval": 8}
    )
    x = torch.tensor([[0.7, 1.3], [1.1, 0.9]], dtype=torch.float64, requires_grad=True)
    time = torch.tensor([[0.2], [0.5]], dtype=torch.float64, requires_grad=True)
    value = torch.exp((pde.sigma**2 + pde.rate) * (1 - time)) * (x**2).sum(
        dim=1, keepdim=True
    )
    dx, dt = torch.autograd.grad(value.sum(), (x, time), create_graph=True)
    dxx = torch.stack(
        [
            torch.autograd.grad(dx[:, j].sum(), x, retain_graph=True)[0][:, j]
            for j in range(2)
        ],
        dim=1,
    )
    z = pde.sigma_matrix(x) * dx
    ito_drift = dt + 0.5 * (pde.sigma_matrix(x).square() * dxx).sum(dim=1, keepdim=True)
    bsde_drift = pde.r_u(time, x, value, z) * value + pde.h_z(time, x, value, z)
    assert_allclose(ito_drift.detach(), bsde_drift.detach(), atol=1e-8)
    assert_allclose(
        pde.terminal(time, x).detach(), (x**2).sum(dim=1, keepdim=True).detach()
    )


def test_default_risk_full_recovery_linear_pricing_limit():
    torch = pytest.importorskip("torch")
    equation = pytest.importorskip("RiskLabAI.pde.equation")
    solver_module = pytest.importorskip("RiskLabAI.pde.solver")
    pde = equation.PricingDefaultRisk(
        {"dim": 1, "total_time": 1, "num_time_interval": 4}
    )
    pde.delta = 1.0

    class ExactGradient(torch.nn.Module):
        def forward(self, time, values):
            return pde.sigma * values

    solver = solver_module.FBSDESolver(
        pde, [1, 4, 1], 0.001, "DTNN", torch.device("cpu")
    )
    solver.solver = ExactGradient()
    dw = torch.tensor(
        [[[0.1, -0.1, 0.2, -0.2]], [[-0.2, 0.1, -0.1, 0.2]]], dtype=torch.float32
    )
    prices = torch.zeros(2, 1, 5)
    prices[:, :, 0] = 100
    for index in range(4):
        prices[:, :, index + 1] = prices[:, :, index] * (
            1 + pde.rate * pde.delta_t + pde.sigma * dw[:, :, index]
        )
    loss, _, _, _ = solver.compute_loss(
        prices, dw, torch.ones(2, 1), torch.tensor([100.0]), torch.tensor([[20.0]])
    )
    assert loss.item() < 1e-8
