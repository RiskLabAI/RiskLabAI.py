"""Single-asset entropic deep hedging with explicit proportional trading costs.

Independent implementation of the mathematical deep-hedging benchmark.
Requires optional PyTorch. All amounts use zero interest and dividend rates;
the supplied GBM drift is the simulation measure's drift, not a pricing claim.
"""

import math

import numpy as np
import torch
from torch import nn

from RiskLabAI.utils._validation import positive_integer, real_scalar

__all__ = [
    "gbm_hedging_paths",
    "hedging_pnl",
    "entropic_risk",
    "black_scholes_call_delta",
    "DeepHedge",
    "train_deep_hedge",
]


def _tensor(value, name, dimensions):
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            f"{name} must be a nonempty finite floating tensor with {dimensions} dimensions."
        )
    return value


def gbm_hedging_paths(*, spot, maturity, volatility, drift, steps, paths, seed):
    """Simulate exact GBM observations with a local CPU float64 generator."""
    spot = real_scalar(spot, "spot", minimum=0, strict=True)
    maturity = real_scalar(maturity, "maturity", minimum=0, strict=True)
    volatility = real_scalar(volatility, "volatility", minimum=0)
    drift = real_scalar(drift, "drift")
    steps, paths = positive_integer(steps, "steps"), positive_integer(paths, "paths")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dt = maturity / steps
    increments = (drift - volatility**2 / 2) * dt + volatility * math.sqrt(
        dt
    ) * torch.randn(paths, steps, generator=generator, dtype=torch.float64)
    result = spot * torch.exp(
        torch.cat(
            [torch.zeros(paths, 1, dtype=torch.float64), increments.cumsum(dim=1)],
            dim=1,
        )
    )
    if not torch.isfinite(result).all() or torch.any(result <= 0):
        raise ValueError("GBM paths exceeded the supported numerical range.")
    return result


def hedging_pnl(prices, positions, *, strike, cost_rate, premium=0.0):
    """Return terminal cash flow of a short call and self-financing hedge.

    Prices are (paths, steps+1); positions are (paths, steps). Initial buying
    and terminal liquidation both pay cost_rate times absolute traded value.
    The caller is responsible for positions using only contemporaneous data.
    """
    prices, positions = _tensor(prices, "prices", 2), _tensor(positions, "positions", 2)
    strike = real_scalar(strike, "strike", minimum=0, strict=True)
    cost = real_scalar(cost_rate, "cost_rate", minimum=0)
    premium = real_scalar(premium, "premium")
    if (
        prices.shape != (positions.shape[0], positions.shape[1] + 1)
        or torch.any(prices <= 0)
        or prices.dtype != positions.dtype
        or prices.device != positions.device
    ):
        raise ValueError(
            "Positive prices and matching positions must share dtype/device."
        )
    zero = torch.zeros_like(positions[:, :1])
    trades = torch.diff(torch.cat([zero, positions, zero], dim=1), dim=1)
    gains = (positions * torch.diff(prices, dim=1)).sum(dim=1)
    costs = cost * (prices * trades.abs()).sum(dim=1)
    result = premium + gains - costs - torch.relu(prices[:, -1] - strike)
    if not torch.isfinite(result).all():
        raise ValueError("Hedging cash flow exceeded the supported numerical range.")
    return result


def entropic_risk(pnl, risk_aversion):
    """Return log(mean(exp(-risk_aversion*pnl)))/risk_aversion stably."""
    pnl = _tensor(pnl, "pnl", 1)
    aversion = real_scalar(risk_aversion, "risk_aversion", minimum=0, strict=True)
    anchor = pnl.min().detach()
    result = (
        -anchor
        + (torch.logsumexp(-aversion * (pnl - anchor), dim=0) - math.log(len(pnl)))
        / aversion
    )
    if not torch.isfinite(result):
        raise ValueError("Entropic risk exceeded the supported numerical range.")
    return result


def black_scholes_call_delta(prices, *, strike, maturity, volatility):
    """Return zero-rate call deltas at all preterminal equally spaced times."""
    prices = _tensor(prices, "prices", 2)
    strike = real_scalar(strike, "strike", minimum=0, strict=True)
    maturity = real_scalar(maturity, "maturity", minimum=0, strict=True)
    volatility = real_scalar(volatility, "volatility", minimum=0, strict=True)
    if prices.shape[1] < 2 or torch.any(prices <= 0):
        raise ValueError("At least two positive price observations are required.")
    steps = prices.shape[1] - 1
    remaining = (
        maturity
        * torch.arange(steps, 0, -1, dtype=prices.dtype, device=prices.device)
        / steps
    )
    d1 = (torch.log(prices[:, :-1] / strike) + volatility**2 * remaining / 2) / (
        volatility * torch.sqrt(remaining)
    )
    return torch.special.ndtr(d1)


class DeepHedge(nn.Module):
    """Shared feedforward hedge using time, current log-price and past holding.

    Positions are unconstrained real values. Initialize with a local CPU seed;
    module initialization preserves the caller's random-number state.
    """

    def __init__(self, hidden_size=16, *, seed=0):
        super().__init__()
        hidden_size = positive_integer(hidden_size, "hidden_size")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            ).double()

    def forward(self, prices, *, strike, maturity):
        """Compute adapted positions; future prices never enter earlier states."""
        prices = _tensor(prices, "prices", 2)
        strike = real_scalar(strike, "strike", minimum=0, strict=True)
        maturity = real_scalar(maturity, "maturity", minimum=0, strict=True)
        if prices.shape[1] < 2 or torch.any(prices <= 0):
            raise ValueError("At least two positive price observations are required.")
        parameter = next(self.parameters())
        if prices.dtype != parameter.dtype or prices.device != parameter.device:
            raise ValueError("Prices must match the model dtype and device.")
        steps = prices.shape[1] - 1
        previous = torch.zeros_like(prices[:, 0])
        positions = []
        for index in range(steps):
            features = torch.stack(
                [
                    torch.full_like(previous, maturity * (steps - index) / steps),
                    torch.log(prices[:, index] / strike),
                    previous,
                ],
                dim=1,
            )
            previous = self.network(features).squeeze(-1)
            positions.append(previous)
        return torch.stack(positions, dim=1)


def train_deep_hedge(
    model,
    training_prices,
    *,
    strike,
    maturity,
    cost_rate,
    risk_aversion,
    iterations,
    learning_rate=0.001,
):
    """Fit only supplied training paths and return per-iteration entropic loss.

    Evaluate on independent paths afterwards. No evaluation outcomes select
    weights or stopping time here; no optimality or superiority is asserted.
    """
    iterations = positive_integer(iterations, "iterations")
    learning_rate = real_scalar(learning_rate, "learning_rate", minimum=0, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(iterations):
        optimizer.zero_grad()
        positions = model(training_prices, strike=strike, maturity=maturity)
        risk = entropic_risk(
            hedging_pnl(training_prices, positions, strike=strike, cost_rate=cost_rate),
            risk_aversion,
        )
        risk.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise RuntimeError("Nonfinite hedging gradients.")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite fitted hedge parameters.")
        history.append(risk.item())
    return np.array(history)
