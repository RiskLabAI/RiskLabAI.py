"""Finite-grid benchmarks for constrained neural option dynamics.

Uses the affine price-state representation and hard boundary restrictions of
Cohen, Reisinger and Wang, with a corrected diffusion transformation in fixed
state coordinates and linear boundary vanishing. This transformation differs
from the paper's literal one-sided, square-root shrinking construction.
These factor models are not a general
surface calibration engine. Static grid inequalities do not certify dynamic
HJM consistency, a risk-neutral pricing measure, or off-grid interpolation.
Requires optional PyTorch. No reference implementation source is incorporated.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from RiskLabAI.utils._validation import positive_integer, real_array, real_scalar

__all__ = [
    "normalized_call_grid_constraints",
    "normalized_call_grid_witness",
    "affine_price_state_interval",
    "IntervalNeuralSDE",
    "fit_interval_neural_sde",
    "simulate_interval_neural_sde",
    "PolytopeNeuralSDE",
    "fit_polytope_neural_sde",
    "simulate_polytope_neural_sde",
    "hjm_drift_residual",
]


def normalized_call_grid_constraints(strikes, maturities):
    """Build weak linear call-price inequalities A*c >= b on a rectangular grid.

    Strikes are positive normalized strikes K/forward; prices are calls divided
    by discounted forward. Flatten prices in maturity-major order. Conditions
    impose intrinsic/upper bounds, vertical-spread slopes in [-1,0], strike
    convexity (including the known strike-zero call price 1), and monotonicity
    in maturity. These weak finite-grid checks alone do not establish an
    attainable full probability distribution in degenerate tail-equality
    cases, or arbitrage freedom outside the supplied grid.
    """
    strikes = real_array(strikes, "strikes", 1)
    maturities = real_array(maturities, "maturities", 1)
    if (
        len(strikes) < 2
        or np.any(strikes <= 0)
        or np.any(maturities <= 0)
        or np.any(np.diff(strikes) <= 0)
        or np.any(np.diff(maturities) <= 0)
    ):
        raise ValueError(
            "Require increasing positive strikes (at least two) and maturities."
        )
    n, m = len(strikes), len(maturities)
    rows, bounds = [], []
    identity = np.eye(n * m)
    for t in range(m):
        for k in range(n):
            e = identity[t * n + k]
            rows.extend([e, -e])
            bounds.extend([max(1 - strikes[k], 0), -1])
        slopes, constants = [], []
        slopes.append(identity[t * n] / strikes[0])
        constants.append(-1 / strikes[0])
        for k in range(n - 1):
            slopes.append(
                (identity[t * n + k + 1] - identity[t * n + k])
                / (strikes[k + 1] - strikes[k])
            )
            constants.append(0)
        for slope, constant in zip(slopes, constants):
            rows.extend([slope, -slope])
            bounds.extend([-1 - constant, constant])
        for k in range(len(slopes) - 1):
            rows.append(slopes[k + 1] - slopes[k])
            bounds.append(constants[k] - constants[k + 1])
        if t:
            for k in range(n):
                rows.append(identity[t * n + k] - identity[(t - 1) * n + k])
                bounds.append(0)
    return np.array(rows), np.array(bounds)


def normalized_call_grid_witness(strikes, maturities, prices):
    """Construct finite probability marginals reproducing a normalized call grid.

    Prices have shape (maturities, strikes). Piecewise linear call curves join
    (0,1), the supplied quotes, and a common zero-price tail strike. Slope
    differences give nonnegative masses with mean one. Calendar ordering at
    their common knots implies ordering of every interpolated call value.
    Reject a positive flat tail, which weak grid inequalities alone admit.
    Tiny negative masses from floating-point arithmetic are rounded to zero;
    all mass, mean and repricing residuals are then checked at 1e-10 tolerance.
    This is a static finite-grid witness, not a calibrated dynamic measure,
    a unique interpolation rule, or evidence about unquoted market prices.
    """
    matrix, bounds = normalized_call_grid_constraints(strikes, maturities)
    strikes = real_array(strikes, "strikes", 1)
    maturities = real_array(maturities, "maturities", 1)
    prices = real_array(prices, "prices", 2)
    if prices.shape != (len(maturities), len(strikes)):
        raise ValueError("Prices must have shape (maturities, strikes).")
    if np.min(matrix @ prices.ravel() - bounds) < -1e-12:
        raise ValueError("Prices violate normalized call-grid inequalities.")
    final_slope = (prices[:, -1] - prices[:, -2]) / (strikes[-1] - strikes[-2])
    positive_tail = prices[:, -1] > 0
    if np.any(positive_tail & (final_slope >= 0)):
        raise ValueError("A positive flat tail has no attainable call-price witness.")
    tail_length = np.zeros(len(maturities))
    tail_length[positive_tail] = prices[positive_tail, -1] / -final_slope[positive_tail]
    tail = strikes[-1] + max(1.0, 2 * float(tail_length.max()))
    if not np.isfinite(tail) or tail <= strikes[-1]:
        raise ValueError("Tail support is not numerically representable.")
    support = np.r_[0, strikes, tail]
    values = np.column_stack(
        [np.ones(len(maturities)), prices, np.zeros(len(maturities))]
    )
    slopes = np.diff(values, axis=1) / np.diff(support)
    probabilities = np.column_stack(
        [1 + slopes[:, 0], np.diff(slopes, axis=1), -slopes[:, -1]]
    )
    if np.min(probabilities) < -1e-12:
        raise ValueError("Call slopes do not yield nonnegative probability masses.")
    probabilities = np.maximum(probabilities, 0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    reconstructed = probabilities @ np.maximum(support[:, None] - strikes, 0)
    mean_error = np.max(np.abs(probabilities @ support - 1))
    price_error = np.max(np.abs(reconstructed - prices))
    if (
        not np.isfinite([mean_error, price_error]).all()
        or max(mean_error, price_error) > 1e-10
    ):
        raise ValueError("Probability witness fails mean or repricing validation.")
    return {
        "support": support,
        "probabilities": probabilities,
        "maximum_mean_error": float(mean_error),
        "maximum_price_error": float(price_error),
    }


def affine_price_state_interval(offset, loading, matrix, bounds):
    """Solve A*(offset+loading*x) >= b exactly for a bounded scalar interval.

    The caller supplies the price basis and complete desired constraints.
    Reject empty, singleton and unbounded state spaces; never repair prices.
    Bounds are linear-algebra results, not an inferred pricing model.
    """
    offset, loading = real_array(offset, "offset", 1), real_array(loading, "loading", 1)
    matrix, bounds = real_array(matrix, "matrix", 2), real_array(bounds, "bounds", 1)
    if offset.shape != loading.shape or matrix.shape != (len(bounds), len(offset)):
        raise ValueError("Price basis and inequality shapes must agree.")
    slopes = matrix @ loading
    right = bounds - matrix @ offset
    if np.any((slopes == 0) & (right > 0)):
        raise ValueError("Price constraints are infeasible for this basis.")
    lower = max(right[slopes > 0] / slopes[slopes > 0], default=-np.inf)
    upper = min(right[slopes < 0] / slopes[slopes < 0], default=np.inf)
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError(
            "The affine price state must have a bounded nonempty interior."
        )
    return float(lower), float(upper)


class IntervalNeuralSDE(nn.Module):
    """Hard-constrained scalar-factor neural SDE on a supplied interval.

    Drift corrections follow the paper's inward correction, with numerator
    (-normal*raw_drift-distance_tolerance)_+. Diffusion amplitude shrinks by
    distance/(diffusion_scale+distance) at the nearest boundary. This is the
    scalar version of the corrected fixed-coordinate polytope transformation,
    with Lipschitz linear vanishing instead of the paper's square-root scale.
    At either boundary diffusion is exactly zero and drift points inward.
    General multi-factor polytope geometry and HJM calibration are excluded.
    """

    def __init__(
        self, lower, upper, *, drift_bound, diffusion_scale, hidden_size=12, seed=0
    ):
        super().__init__()
        self.lower = real_scalar(lower, "lower")
        self.upper = real_scalar(upper, "upper")
        self.drift_bound = real_scalar(
            drift_bound, "drift_bound", minimum=0, strict=True
        )
        self.diffusion_scale = real_scalar(
            diffusion_scale, "diffusion_scale", minimum=0, strict=True
        )
        hidden_size = positive_integer(hidden_size, "hidden_size")
        if self.lower >= self.upper or not np.isfinite(self.upper - self.lower):
            raise ValueError("Interval width must be finite and positive.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(1, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2)
            ).double()

    def forward(self, state):
        """Return drift and diffusion for a finite vector of admissible states."""
        parameter = next(self.parameters())
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 1
            or state.numel() == 0
            or state.dtype != parameter.dtype
            or state.device != parameter.device
            or not torch.isfinite(state).all()
            or torch.any((state < self.lower) | (state > self.upper))
        ):
            raise ValueError(
                "States must be a finite admissible vector matching model dtype/device."
            )
        width = self.upper - self.lower
        center = self.lower + width / 2
        raw = self.network(((state - center) / width).unsqueeze(-1))
        raw_drift = self.drift_bound * torch.tanh(raw[:, 0])
        raw_diffusion = F.softplus(raw[:, 1])
        rho = width / 4
        left, right = state - self.lower, self.upper - state
        left_correction = F.relu(-raw_drift - self.drift_bound * left / rho)
        right_correction = F.relu(raw_drift - self.drift_bound * right / rho)
        drift = raw_drift + left_correction - right_correction
        distance = torch.minimum(left, right)
        shrinking = distance / (self.diffusion_scale + distance)
        diffusion = raw_diffusion * shrinking
        return drift, diffusion

    def negative_log_likelihood(self, states, next_states, dt):
        """Return mean Gaussian Euler transition NLL for strictly interior starts.

        This approximate likelihood is not the exact constrained transition
        density. Boundary starts have singular diffusion and are rejected;
        adding variance jitter would violate the hard boundary contract.
        """
        dt = real_scalar(dt, "dt", minimum=0, strict=True)
        drift, diffusion = self(states)
        if (
            not isinstance(next_states, torch.Tensor)
            or next_states.shape != states.shape
            or next_states.dtype != states.dtype
            or next_states.device != states.device
            or not torch.isfinite(next_states).all()
            or torch.any((next_states < self.lower) | (next_states > self.upper))
            or torch.any(diffusion <= 0)
        ):
            raise ValueError(
                "Likelihood requires interior starts and matching admissible next states."
            )
        variance = diffusion.square() * dt
        loss = (
            0.5
            * (
                math.log(2 * math.pi)
                + torch.log(variance)
                + (next_states - states - drift * dt).square() / variance
            ).mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("Transition likelihood is nonfinite.")
        return loss


def fit_interval_neural_sde(
    model, states, next_states, *, dt, iterations, learning_rate=0.001
):
    """Fit Euler likelihood on supplied training transitions only."""
    iterations = positive_integer(iterations, "iterations")
    learning_rate = real_scalar(learning_rate, "learning_rate", minimum=0, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(states, next_states, dt)
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise RuntimeError("Nonfinite neural-SDE gradients.")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite neural-SDE parameters.")
        history.append(loss.item())
    return np.array(history)


def simulate_interval_neural_sde(model, initial, brownian_increments, *, dt):
    """Apply ordinary Euler steps to caller-supplied Brownian increments.

    Continuous-time boundary conditions do not make Euler steps invariant.
    Raise on any exit, reporting the step; never clip, reflect, discard paths
    or claim a price from the failed run. Use coupled increments at finer
    steps for sensitivity checks. The simulation is under the fitted measure.
    """
    dt = real_scalar(dt, "dt", minimum=0, strict=True)
    model(initial)
    if (
        not isinstance(brownian_increments, torch.Tensor)
        or brownian_increments.ndim != 2
        or brownian_increments.shape[0] != len(initial)
        or brownian_increments.shape[1] == 0
        or brownian_increments.dtype != initial.dtype
        or brownian_increments.device != initial.device
        or not torch.isfinite(brownian_increments).all()
    ):
        raise ValueError(
            "Brownian increments must match initial states and model dtype/device."
        )
    states = [initial.detach().clone()]
    with torch.no_grad():
        for index in range(brownian_increments.shape[1]):
            drift, diffusion = model(states[-1])
            proposal = (
                states[-1] + drift * dt + diffusion * brownian_increments[:, index]
            )
            if not torch.isfinite(proposal).all() or torch.any(
                (proposal < model.lower) | (proposal > model.upper)
            ):
                raise RuntimeError(
                    f"Euler step {index + 1} exits the admissible price state; reduce the step and reassess."
                )
            states.append(proposal)
    return torch.stack(states, dim=1)


class PolytopeNeuralSDE(nn.Module):
    """Neural SDE with corrected continuous polytope diffusion shrinking.

    Supply V,b for V*x>=b and a strictly interior point. Price-state constraints
    are obtained from A*(offset+basis*x)>=price_bounds. The caller supplies the
    grid, price basis and constraints; this class does not identify factors.
    Normal rows are normalized internally without changing the feasible set.
    For an orthonormal column frame Q from nearest independent normals, use
    sigma = Q diag(g(distances)) Q.T L, where g(d)=d/(diffusion_scale+d) and L
    is the neural base diffusion. Both coordinate transforms are required:
    equal-distance frame permutations then leave the shrinking matrix intact.
    This corrects the paper's literal one-sided transformation; linear rather
    than square-root vanishing makes coefficients Lipschitz at a face.
    Drift corrections point toward
    the supplied interior point and vanish beyond a fixed interior distance.
    Negative distances within a relative 1e-12 roundoff tolerance are evaluated
    as zero; positive distances are never snapped to zero. Normal geometry must
    be numerically well conditioned. HJM consistency, an attainable price grid
    and the fitted measure require separate checks. Boundary invariance does
    not certify dynamic arbitrage freedom or invariant discrete Euler steps.
    """

    def __init__(
        self,
        normals,
        bounds,
        interior,
        *,
        drift_bound,
        diffusion_scale,
        hidden_size=16,
        seed=0,
    ):
        super().__init__()
        from scipy.optimize import linprog

        normals = real_array(normals, "normals", 2)
        bounds, interior = real_array(bounds, "bounds", 1), real_array(
            interior, "interior", 1
        )
        self.dimension = len(interior)
        if normals.shape != (len(bounds), self.dimension):
            raise ValueError("Normals, bounds and interior dimensions must agree.")
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms == 0) or np.linalg.matrix_rank(normals) < self.dimension:
            raise ValueError("Normals must be nonzero and span the state dimension.")
        normals, bounds = normals / norms[:, None], bounds / norms
        slack = normals @ interior - bounds
        if np.any(slack <= 0):
            raise ValueError(
                "The supplied interior must satisfy every inequality strictly."
            )
        for direction in np.r_[np.eye(self.dimension), -np.eye(self.dimension)]:
            result = linprog(
                direction,
                A_ub=-normals,
                b_ub=-bounds,
                bounds=[(None, None)] * self.dimension,
                method="highs",
            )
            if not result.success:
                raise ValueError(
                    "The supplied polytope must be bounded and numerically feasible."
                )
        self.drift_bound = real_scalar(
            drift_bound, "drift_bound", minimum=0, strict=True
        )
        self.diffusion_scale = real_scalar(
            diffusion_scale, "diffusion_scale", minimum=0, strict=True
        )
        self.rho = float(slack.min() / 2)
        self.register_buffer("normals", torch.tensor(normals, dtype=torch.float64))
        self.register_buffer("bounds", torch.tensor(bounds, dtype=torch.float64))
        self.register_buffer("interior", torch.tensor(interior, dtype=torch.float64))
        hidden_size = positive_integer(hidden_size, "hidden_size")
        outputs = self.dimension + self.dimension * (self.dimension + 1) // 2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(self.dimension, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, outputs),
            ).double()

    def _distances(self, states):
        if (
            not isinstance(states, torch.Tensor)
            or states.ndim != 2
            or states.shape[1] != self.dimension
            or states.shape[0] == 0
            or states.dtype != self.normals.dtype
            or states.device != self.normals.device
            or not torch.isfinite(states).all()
        ):
            raise ValueError(
                "States must be finite rows matching the polytope dtype/device."
            )
        distances = states @ self.normals.T - self.bounds
        tolerance = 1e-12 * max(
            1.0, self.bounds.abs().max().item(), states.abs().max().item()
        )
        if torch.any(distances < -tolerance):
            raise ValueError("State lies outside the supplied polytope.")
        return distances.clamp_min(0)

    def forward(self, states):
        """Return drift (rows,d) and diffusion (rows,d,d), with hard boundaries."""
        distances = self._distances(states)
        raw = self.network((states - self.interior) / self.rho)
        drift = self.drift_bound * torch.tanh(raw[:, : self.dimension])
        triangular = torch.tril_indices(
            self.dimension, self.dimension, device=states.device
        )
        entries = raw[:, self.dimension :]
        entries = torch.where(
            triangular[0] == triangular[1], F.softplus(entries), entries
        )
        base_diffusion = torch.zeros(
            len(states),
            self.dimension,
            self.dimension,
            dtype=states.dtype,
            device=states.device,
        )
        base_diffusion[:, triangular[0], triangular[1]] = entries
        transformed = []
        for index, distance in enumerate(distances):
            directions, scales = [], []
            for face in torch.argsort(distance.detach(), stable=True).tolist():
                normal = self.normals[face]
                residual = normal
                for _ in range(2):
                    for direction in directions:
                        residual = residual - torch.dot(residual, direction) * direction
                norm = torch.linalg.vector_norm(residual)
                if norm.item() > 1e-10:
                    directions.append(residual / norm)
                    scales.append(
                        distance[face] / (self.diffusion_scale + distance[face])
                    )
                if len(directions) == self.dimension:
                    break
            if len(directions) != self.dimension:
                raise RuntimeError("Boundary normals are numerically rank deficient.")
            basis = torch.stack(directions, dim=1)
            shrinking = (basis * torch.stack(scales)) @ basis.T
            transformed.append(shrinking @ base_diffusion[index])
        corrections = self.interior - states
        normal_drift = drift @ self.normals.T
        inward_product = corrections @ self.normals.T
        active = distances < self.rho
        safe_denominator = torch.where(
            active, inward_product, torch.ones_like(inward_product)
        )
        tolerance = self.drift_bound * math.sqrt(self.dimension) * distances / self.rho
        weights = torch.where(
            active,
            F.relu(-normal_drift - tolerance) / safe_denominator,
            torch.zeros_like(distances),
        )
        drift = drift + weights.sum(dim=1, keepdim=True) * corrections
        return drift, torch.stack(transformed)

    def negative_log_likelihood(self, states, next_states, dt):
        """Mean approximate Euler Gaussian NLL; starts must be strictly interior."""
        dt = real_scalar(dt, "dt", minimum=0, strict=True)
        if torch.any(self._distances(states) <= 0):
            raise ValueError("Likelihood starts must be strictly interior.")
        self._distances(next_states)
        if next_states.shape != states.shape:
            raise ValueError("Transition shapes must match.")
        drift, diffusion = self(states)
        covariance = dt * diffusion @ diffusion.transpose(-1, -2)
        factor, info = torch.linalg.cholesky_ex(covariance)
        if torch.any(info != 0):
            raise RuntimeError(
                "Transition covariance is not numerically positive definite."
            )
        residual = (next_states - states - dt * drift).unsqueeze(-1)
        standardized = torch.linalg.solve_triangular(factor, residual, upper=False)
        log_determinant = 2 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(
            dim=1
        )
        loss = (
            0.5
            * (
                self.dimension * math.log(2 * math.pi)
                + log_determinant
                + standardized.square().sum(dim=(1, 2))
            ).mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("Transition likelihood is nonfinite.")
        return loss


def hjm_drift_residual(price_basis, drift, diffusion, required_price_drift):
    """Report compatibility of the supplied finite-grid HJM linear equation.

    Solve (G*sigma)*market_price_of_risk = G*mu - required_price_drift by
    least squares. G has (prices,factors); the required drift is independently
    supplied from the paper's maturity/moneyness derivatives and stock model.
    A nonzero residual signals incompatibility. Even zero residual alone does
    not establish an equivalent martingale measure or verify those derivatives.
    """
    basis, diffusion = real_array(price_basis, "price_basis", 2), real_array(
        diffusion, "diffusion", 2
    )
    drift = real_array(drift, "drift", 1)
    required = real_array(required_price_drift, "required_price_drift", 1)
    if basis.shape != (len(required), len(drift)) or diffusion.shape != (
        len(drift),
        len(drift),
    ):
        raise ValueError("Price and state drift/diffusion dimensions must agree.")
    operator = basis @ diffusion
    rhs = basis @ drift - required
    risk_price, _, rank, singular = np.linalg.lstsq(operator, rhs, rcond=None)
    residual = rhs - operator @ risk_price
    return {
        "market_price_of_risk": risk_price,
        "residual": residual,
        "squared_residual": float(residual @ residual),
        "rank": int(rank),
        "singular_values": singular,
    }


def fit_polytope_neural_sde(
    model, states, next_states, *, dt, iterations, learning_rate=0.001
):
    """Fit supplied interior training transitions by approximate Euler likelihood.

    Training uses no validation observations. Finite optimization does not
    certify parameter recovery, HJM consistency or global optimality.
    """
    return fit_interval_neural_sde(
        model,
        states,
        next_states,
        dt=dt,
        iterations=iterations,
        learning_rate=learning_rate,
    )


def simulate_polytope_neural_sde(model, initial, brownian_increments, *, dt):
    """Euler simulation with explicit exit failure and no projection or clipping.

    initial is (paths,factors); increments are (paths,steps,factors), with
    Brownian variance dt supplied by the caller. A failed path invalidates the
    entire run. Compare coupled finer increments to assess discretization.
    """
    dt = real_scalar(dt, "dt", minimum=0, strict=True)
    model._distances(initial)
    if (
        not isinstance(brownian_increments, torch.Tensor)
        or brownian_increments.ndim != 3
        or brownian_increments.shape[0] != len(initial)
        or brownian_increments.shape[1] == 0
        or brownian_increments.shape[2] != model.dimension
        or brownian_increments.dtype != initial.dtype
        or brownian_increments.device != initial.device
        or not torch.isfinite(brownian_increments).all()
    ):
        raise ValueError(
            "Brownian increments must match state dimensions and dtype/device."
        )
    states = [initial.detach().clone()]
    with torch.no_grad():
        for index in range(brownian_increments.shape[1]):
            drift, diffusion = model(states[-1])
            proposal = (
                states[-1]
                + drift * dt
                + (diffusion @ brownian_increments[:, index, :, None]).squeeze(-1)
            )
            try:
                model._distances(proposal)
            except ValueError as error:
                raise RuntimeError(
                    f"Euler step {index + 1} exits the polytope; reduce the step and reassess."
                ) from error
            states.append(proposal)
    return torch.stack(states, dim=1)
