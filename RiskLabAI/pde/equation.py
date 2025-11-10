"""
Base class and concrete implementations for Partial Differential Equations
(PDEs) to be solved by the Deep BSDE solver.

Based on: https://github.com/frankhan91/DeepBSDE
"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Tuple, Optional, Union

class Equation:
    """
    Base class for defining PDE-related functions.
    """

    def __init__(self, eqn_config: dict):
        """
        Initialize the PDE configuration.

        Parameters
        ----------
        eqn_config : dict
            Dictionary containing PDE parameters:
            - 'dim': Dimensionality of the problem (int)
            - 'total_time': Total time horizon (float)
            - 'num_time_interval': Number of time steps (int)
        """
        self.dim: int = eqn_config['dim']
        self.total_time: float = eqn_config['total_time']
        self.num_time_interval: int = eqn_config['num_time_interval']
        self.delta_t: float = self.total_time / self.num_time_interval
        self.sqrt_delta_t: float = np.sqrt(self.delta_t)
        self.y_init: Optional[float] = None

    def sample(self, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the forward SDE (e.g., the underlying asset path).

        Parameters
        ----------
        num_sample : int
            Number of sample paths to generate.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - dw_sample: Wiener process increments [num_sample, dim, num_time_interval]
            - x_sample: Asset paths [num_sample, dim, num_time_interval+1]
        """
        raise NotImplementedError

    def r_u(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        The driver function 'f' of the BSDE, or the generator of the PDE.
        Represents the non-linear interest rate and dividend term.

        Parameters
        ----------
        t : float
            Current time.
        x : Tensor
            Space coordinates [batch_size, dim].
        y : Tensor
            Function value (price) [batch_size, 1].
        z : Tensor
            Gradient (delta) [batch_size, dim].

        Returns
        -------
        Tensor
            Value of the driver `f` [batch_size, 1].
        """
        raise NotImplementedError

    def h_z(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        The Hamiltonian term H(z) in the PDE, or the volatility-related
        term in the BSDE driver.

        Parameters
        ----------
        t : float
            Current time.
        x : Tensor
            Space coordinates [batch_size, dim].
        y : Tensor
            Function value (price) [batch_size, 1].
        z : Tensor
            Gradient (delta) [batch_size, dim].

        Returns
        -------
        Tensor
            Value of the Hamiltonian `H(z)` [batch_size, 1].
        """
        raise NotImplementedError

    def terminal(self, t: float, x: Tensor) -> Tensor:
        """
        Terminal condition (payoff) of the PDE.

        Parameters
        ----------
        t : float
            Current time (should be total_time).
        x : Tensor
            Space coordinates [batch_size, dim].

        Returns
        -------
        Tensor
            Terminal payoff value [batch_size, 1].
        """
        raise NotImplementedError
        
    def sigma_matrix(self, x: Tensor) -> Tensor:
        """Helper to get the volatility matrix sigma(x)."""
        raise NotImplementedError
        
    def terminal_for_sample(self, x: Tensor) -> Tensor:
        """Terminal condition for a multi-sample path."""
        raise NotImplementedError


class PricingDefaultRisk(Equation):
    """
    PDE for pricing with default risk.
    """
    def __init__(self, eqn_config: dict):
        super(PricingDefaultRisk, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02   # R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.K = 100
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        dw_sample = np.random.normal(
            size=[num_sample, self.dim, self.num_time_interval]
        ) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[
                :, :, i
            ] + (self.sigma_matrix(x_sample[:, :, i]) * dw_sample[:, :, i])
        return dw_sample, x_sample

    def r_u(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        piecewise_linear = nn.ReLU()(
            nn.ReLU()(y - self.vh) * self.slope + self.gammah - self.gammal
        ) + self.gammal
        return (1 - self.delta) * piecewise_linear + self.rate

    def h_z(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        return torch.zeros((x.size()[0], 1), device=x.device)

    def sigma_matrix(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        return self.sigma * x

    def terminal(self, t: float, x: Tensor) -> Tensor:
        # torch.min(x, 1) returns (values, indices)
        return nn.ReLU()(torch.min(x, 1, keepdim=True)[0])

    def terminal_for_sample(self, x: Tensor) -> Tensor:
        return nn.ReLU()(torch.min(x, 2, keepdim=True)[0])


class HJBLQ(Equation):
    """
    Hamilton-Jacobi-Bellman (HJB) equation with Linear-Quadratic (LQ) cost.
    """
    def __init__(self, eqn_config: dict):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        dw_sample = np.random.normal(
            size=[num_sample, self.dim, self.num_time_interval]
        ) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def r_u(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        return torch.zeros((x.size()[0], 1), device=x.device)

    def h_z(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        return torch.sum(torch.square(z), dim=1, keepdim=True) / (self.sigma**2)

    def terminal(self, t: float, x: Tensor) -> Tensor:
        return torch.log(0.5 * (1 + torch.norm(x, p=2, dim=1, keepdim=True)**2))

    def sigma_matrix(self, x: Union[np.ndarray, Tensor]) -> float:
        return self.sigma
        
    def terminal_for_sample(self, x: Tensor) -> Tensor:
        # Used by Monte-Carlo solver, needs to match HJB terminal
        return torch.log(0.5 * (1 + torch.norm(x, p=2, dim=2, keepdim=True)**2))


class BlackScholesBarenblatt(Equation):
    """
    Black-Scholes-Barenblatt equation.
    """
    def __init__(self, eqn_config: dict):
        super(BlackScholesBarenblatt, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * np.array([1.0 / (1.0 + i % 2) for i in range(self.dim)])
        self.sigma = 0.4
        self.rate = 0.05   # interest rate R
        self.mu_bar = 0.0

    def sample(self, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        dw_sample = np.random.normal(
            size=(num_sample, self.dim, self.num_time_interval)
        ) * self.sqrt_delta_t
        x_sample = np.zeros((num_sample, self.dim, self.num_time_interval + 1))
        x_sample[:, :, 0] = np.ones((num_sample, self.dim)) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[
                :, :, i
            ] + (self.sigma_matrix(x_sample[:, :, i]) * dw_sample[:, :, i])
        return dw_sample, x_sample

    def r_u(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        return torch.ones((x.size()[0], 1), device=x.device) * self.rate

    def h_z(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        return -1 * torch.sum(z, dim=1, keepdim=True) * self.rate / self.sigma

    def terminal(self, t: float, x: Tensor) -> Tensor:
        return torch.sum(x ** 2, dim=1, keepdim=True)

    def sigma_matrix(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        return self.sigma * x

    def terminal_for_sample(self, x: Tensor) -> Tensor:
        return torch.sum(x ** 2, dim=2, keepdim=True)


class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes with different interest rates for
    borrowing and lending.
    """
    def __init__(self, eqn_config: dict):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        dw_sample = np.random.normal(
            size=[num_sample, self.dim, self.num_time_interval]
        ) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.mu_bar - (self.sigma**2) / 2) * self.delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (
                factor * np.exp(self.sigma * dw_sample[:, :, i])
            ) * x_sample[:, :, i]
        return dw_sample, x_sample

    def r_u(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        temp = torch.sum(z, dim=1, keepdim=True) / self.sigma - y
        return torch.where(temp > 0, self.rb, self.rl)

    def h_z(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        temp = torch.sum(z, dim=1, keepdim=True) / self.sigma - y
        sum_z = torch.sum(z, dim=1, keepdim=True)
        return torch.where(
            temp > 0,
            (self.mu_bar - self.rb) * sum_z / self.sigma,
            (self.mu_bar - self.rl) * sum_z / self.sigma,
        )

    def terminal(self, t: float, x: Tensor) -> Tensor:
        temp = torch.max(x, 1, keepdim=True)[0]
        return nn.ReLU()(temp - 120) - 2 * nn.ReLU()(temp - 150)

    def sigma_matrix(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        return self.sigma * x

    def terminal_for_sample(self, x: Tensor) -> Tensor:
        temp = torch.max(x, 2, keepdim=True)[0]
        return torch.maximum(temp - 120, torch.tensor(0.0, device=x.device))