"""
Tests for the pde/ module (Deep BSDE Solver)
"""

import pytest
torch = pytest.importorskip("torch")
import torch
import numpy as np

# Import the main components
from RiskLabAI.pde.equation import HJBLQ
from RiskLabAI.pde.solver import FBSDESolver

@pytest.fixture
def pde_config():
    """Fixture for a simple PDE configuration."""
    return {
        'dim': 1,
        'total_time': 1.0,
        'num_time_interval': 10,
    }

@pytest.fixture
def device():
    """Fixture to determine device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_pde_solver_smoke_test(pde_config, device):
    """
    A "smoke test" to ensure the FBSDESolver can be
    initialized and the solve() method runs for a few iterations
    without crashing.
    """
    # 1. Initialize Equation
    pde = HJBLQ(pde_config)
    
    # 2. Initialize Solver
    layer_sizes = [pde.dim + 1] + [32, 32] + [pde.dim]
    solver = FBSDESolver(
        pde=pde,
        layer_sizes=layer_sizes,
        learning_rate=0.001,
        solving_method='DTNN',
        device=device
    )
    
    # 3. Run solver for a few steps
    num_iterations = 2
    batch_size = 16
    init_y = 0.5
    
    losses, inits = solver.solve(
        num_iterations=num_iterations,
        batch_size=batch_size,
        init_y=init_y
    )
    
    # 4. Check results
    assert isinstance(losses, list)
    assert len(losses) == num_iterations
    assert isinstance(losses[0], float)
    
    assert isinstance(inits, list)
    assert len(inits) == num_iterations
    assert isinstance(inits[0], float)