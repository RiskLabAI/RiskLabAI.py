r"""
Leakage-aware hyper-parameter optimization: Optuna wired through purged CV, gated by the Deflated Sharpe.

de Prado tunes via grid / random search under purged cross-validation. Principled Bayesian / evolutionary
search (Optuna's TPE / CMA-ES) reaches good configurations in far fewer trials, but two hazards make naive
HPO dangerous on financial data: leakage-prone CV (a shuffled k-fold) massively inflates the apparent
score under overlapping labels, and every trial is a configuration, so the search itself overfits and the
in-sample HPO score must be deflated by the trial count before it can be trusted.

This module wires Optuna through the repo's :class:`PurgedKFold` / :class:`CombinatorialPurged` (so the
per-trial score is leakage-controlled) and provides a Deflated-Sharpe gate that deflates the selected
model's out-of-sample Sharpe by the HPO-inclusive trial count.

Admitted in Appraisal 20 (CONTRIBUTIONS_LEDGER 2026-06-27) as a methodology / infrastructure admit, NOT a
performance claim. Scope tag, verbatim from the verdict:

    Optuna (TPE/CMA-ES) wired through PurgedKFold/CPCV with PBO/DSR gating, preferred over grid/random for
    search efficiency and over naive-CV tuning for leakage-safety, and carrying the honest, characterized
    finding that tuning yields no OOS edge after deflation - so it must be gated by the Deflated Sharpe at
    the trial count, never trusted on the in-sample HPO score.

**Plainly: this improves search efficiency and leakage-safety, NOT out-of-sample performance. Tuning does
not create edge.** In the appraisal the principled-HPO real-leg gain did not survive the Deflated Sharpe at
the HPO-inclusive trial count (DSR 0.44 / 0.34 < 0.5, PBO 0.67 / 0.69, OOS at the base rate); the value is
that the search reaches the optimum in fewer trials and that purged CV removes the leakage a naive k-fold
inflates. Always gate the selected model by :func:`deflated_sharpe_gate` at the HPO trial count; never
trust the in-sample HPO score. Evidence and caveats: appraisals/20_verdict.md.

References
----------
Akiba, T. et al. (2019) Optuna: a next-generation hyperparameter optimization framework. KDD.
Lopez de Prado, M. (2018) Advances in Financial Machine Learning (PurgedKFold/CPCV, PBO, the Deflated
    Sharpe Ratio).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from RiskLabAI.backtest.probabilistic_sharpe_ratio import probabilistic_sharpe_ratio
from RiskLabAI.backtest.test_set_overfitting import expected_max_sharpe_ratio

from .purged_kfold import PurgedKFold


def _purged_cv_score(
    estimator_factory, params, x, y, times, n_splits, embargo, scoring, seed
):
    """Mean leakage-controlled CV score of one configuration under PurgedKFold."""
    purged = PurgedKFold(n_splits=n_splits, times=times, embargo=embargo)
    x_values = x.values if hasattr(x, "values") else np.asarray(x)
    y_values = y.values if hasattr(y, "values") else np.asarray(y)
    scores = []
    for train, test in purged.split(x):
        if len(train) < 25 or len(test) < 5 or len(np.unique(y_values[train])) < 2:
            continue
        model = estimator_factory(params)
        model.fit(x_values[train], y_values[train])
        if scoring is None:
            scores.append(
                float((model.predict(x_values[test]) == y_values[test]).mean())
            )
        else:
            scores.append(float(scoring(model, x_values[test], y_values[test])))
    return float(np.mean(scores)) if scores else float("-inf")


def leakage_aware_hpo(
    suggest_params: Callable,
    estimator_factory: Callable,
    x,
    y,
    times,
    n_trials: int = 50,
    sampler: str = "tpe",
    n_splits: int = 5,
    embargo: float = 0.0,
    scoring: Callable | None = None,
    random_state: int = 0,
) -> dict:
    r"""
    Optuna hyper-parameter search with every trial scored under :class:`PurgedKFold` (leakage-controlled).

    This improves SEARCH EFFICIENCY (TPE / CMA-ES reach the optimum in far fewer trials than grid / random)
    and LEAKAGE-SAFETY (the per-trial score is purged, not a leaky shuffled k-fold). It does NOT improve
    out-of-sample performance - tuning does not create edge - so the selected model must be gated by
    :func:`deflated_sharpe_gate` at the returned ``n_trials``; never trust ``best_score`` directly. See the
    module docstring for the verbatim scope tag and appraisals/20_verdict.md.

    Parameters
    ----------
    suggest_params : callable
        ``suggest_params(trial) -> dict``: maps an Optuna trial to a hyper-parameter dict.
    estimator_factory : callable
        ``estimator_factory(params) -> estimator``: builds a fresh scikit-learn-style estimator
        (``fit`` / ``predict``).
    x, y : array-like or pandas
        Design matrix and labels. ``x`` must support ``PurgedKFold.split(x)`` (a DataFrame indexed by the
        event start times).
    times : pd.Series
        Event start -> end times for purging / embargo (passed to :class:`PurgedKFold`).
    n_trials : int, default 50
        Optuna trial budget (this is the HPO trial count to deflate by).
    sampler : {"tpe", "cmaes"}, default "tpe"
        The Optuna sampler.
    n_splits : int, default 5
        Purged CV folds.
    embargo : float, default 0.0
        PurgedKFold embargo fraction.
    scoring : callable, optional
        ``scoring(estimator, x_test, y_test) -> float`` (higher is better). Defaults to accuracy.
    random_state : int, default 0
        Sampler seed.

    Returns
    -------
    dict
        ``best_params``, ``best_score`` (purged CV), ``n_trials``, ``trial_scores``, ``mean_trial_score``,
        ``std_trial_score``.

    Raises
    ------
    ImportError
        If ``optuna`` is not installed (it is an optional dependency).
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "leakage_aware_hpo requires optuna (pip install optuna)"
        ) from exc

    if sampler == "tpe":
        optuna_sampler = optuna.samplers.TPESampler(seed=random_state)
    elif sampler == "cmaes":
        optuna_sampler = optuna.samplers.CmaEsSampler(seed=random_state)
    else:
        raise ValueError("sampler must be 'tpe' or 'cmaes'")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest_params(trial)
        return _purged_cv_score(
            estimator_factory,
            params,
            x,
            y,
            times,
            n_splits,
            embargo,
            scoring,
            random_state,
        )

    study = optuna.create_study(direction="maximize", sampler=optuna_sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trial_scores = [
        t.value for t in study.trials if t.value is not None and np.isfinite(t.value)
    ]
    return {
        "best_params": study.best_params,
        "best_score": float(study.best_value),
        "n_trials": int(n_trials),
        "trial_scores": trial_scores,
        "mean_trial_score": (
            float(np.mean(trial_scores)) if trial_scores else float("nan")
        ),
        "std_trial_score": (
            float(np.std(trial_scores)) if trial_scores else float("nan")
        ),
    }


def deflated_sharpe_gate(
    out_of_sample_returns: np.ndarray,
    n_trials: int,
    trial_sharpe_std: float,
    threshold: float = 0.5,
) -> dict:
    r"""
    Gate a selected model's out-of-sample Sharpe by the Deflated Sharpe at the HPO trial count.

    The benchmark is the expected maximum Sharpe over ``n_trials`` independent trials (with the observed
    cross-trial Sharpe dispersion ``trial_sharpe_std``); the Deflated Sharpe is the probability that the
    realized OOS Sharpe exceeds that benchmark. A selection passes only if the Deflated Sharpe exceeds
    ``threshold`` (0.5). This is the decisive control: in the appraisal the principled-HPO gain did not
    pass it - tuning yielded no out-of-sample edge after deflation. See the module docstring for the
    verbatim scope tag and appraisals/20_verdict.md.

    Parameters
    ----------
    out_of_sample_returns : np.ndarray
        The selected model's out-of-sample (e.g. CPCV) per-period return series.
    n_trials : int
        The HPO-inclusive trial count to deflate by (every configuration tried).
    trial_sharpe_std : float
        The standard deviation of the per-trial (per-period) Sharpe ratios across the search.
    threshold : float, default 0.5
        The Deflated-Sharpe pass threshold.

    Returns
    -------
    dict
        ``observed_sharpe``, ``benchmark_sharpe`` (E[max SR] at ``n_trials``), ``n_trials``,
        ``deflated_sharpe``, and ``passes``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(500) * 0.01
    >>> gate = deflated_sharpe_gate(r, n_trials=100, trial_sharpe_std=0.05)
    >>> set(gate) >= {"deflated_sharpe", "passes", "benchmark_sharpe"}
    True
    """
    from scipy.stats import kurtosis, skew

    r = np.asarray(out_of_sample_returns, dtype=float)
    sd = r.std()
    observed_sharpe = float(r.mean() / sd) if sd > 0 else 0.0
    benchmark_sharpe = float(expected_max_sharpe_ratio(n_trials, 0.0, trial_sharpe_std))
    deflated = float(
        probabilistic_sharpe_ratio(
            observed_sharpe,
            benchmark_sharpe,
            r.size,
            skewness_of_returns=float(skew(r)) if r.size > 2 else 0.0,
            kurtosis_of_returns=float(kurtosis(r, fisher=False)) if r.size > 3 else 3.0,
        )
    )
    return {
        "observed_sharpe": observed_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "n_trials": int(n_trials),
        "deflated_sharpe": deflated,
        "passes": bool(deflated > threshold),
    }
