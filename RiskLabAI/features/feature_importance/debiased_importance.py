r"""
Debiased and conditional feature importance: MDI+ and the Conditional Predictive Impact (CPI).

de Prado's MDI counts impurity reduction, which inflates with split opportunities (high cardinality)
and with in-sample reuse, so it over-credits noisy / continuous features and can rank noise above
signal; MDA has no significance test. Two published fixes extend the baselines in
`features.feature_importance`:

- MDI+ (Agarwal et al. 2023) recasts each tree's MDI as a regularized GLM on the tree's decision-stump
  representation and scores a feature by its out-of-sample partial contribution, removing the
  in-sample / cardinality inflation; and
- the Conditional Predictive Impact (CPI, Watson and Wright 2021) measures a feature's importance as
  the model-loss increase when the true feature is replaced by a conditionally-resampled knockoff, with
  a paired significance test, the calibrated test MDA lacks.

Admitted in Appraisal 10 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tags, verbatim from the verdict:

    MDI+: prefer MDI+ over MDI when features are noisy, high-cardinality, or mixed-type - it rejects
    noise/cardinality inflation (noise-reject 0.94 vs 0.50) and recovers true relevance better;
    converges to MDI when features are orthogonal and high-SNR.

    CPI: prefer CPI when you need a statistically valid test of whether a feature matters (MDA has
    none); it holds nominal size with good power.

The implemented MDI+ is the simplified faithful variant admitted (per-tree decision-stump ridge GLM
with an out-of-bag partial-variance score), NOT the full leave-one-out / similarity-weighted paper
algorithm, which is a logged refinement. CPI's test holds nominal size in the analysis regimes; on the
hardest held-out corner (high correlation x low SNR) it is mildly elevated (~0.067 vs 0.05), a small
finite-sample effect. Evidence and caveats: appraisals/10_verdict.md.

References
----------
Agarwal, A. et al. (2023) MDI+: A flexible random forest based feature importance framework.
    arXiv:2307.01932.
Watson, D. S. and Wright, M. N. (2021) Testing conditional independence in supervised learning
    algorithms. Machine Learning, 110, 2107-2129. (The Conditional Predictive Impact.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

_DEFAULT_N_ESTIMATORS = 100
_DEFAULT_MAX_DEPTH = 6
_DEFAULT_TREES_DECOMP = 60


def _default_forest(
    random_state: int, n_estimators: int = _DEFAULT_N_ESTIMATORS
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=_DEFAULT_MAX_DEPTH,
        random_state=random_state,
        n_jobs=1,
        bootstrap=True,
    )


def mdi_plus_importance(
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int = 0,
    n_trees: int = _DEFAULT_TREES_DECOMP,
    classifier: RandomForestClassifier | None = None,
) -> pd.Series:
    r"""
    MDI+ feature importance (Agarwal et al. 2023, simplified faithful variant).

    For each tree it builds a centered decision-stump basis (one indicator ``1{x_f <= threshold}`` per
    internal node, grouped by the splitting feature), fits a ridge GLM of the label on that basis using
    the tree's in-bag rows, and scores each feature by the variance of its block's partial prediction
    evaluated on the tree's out-of-bag rows; the per-tree scores are averaged. The tree representation
    plus regularized GLM plus out-of-sample partial contribution remove the in-sample / cardinality
    inflation of plain MDI.

    Prefer MDI+ over MDI when features are noisy, high-cardinality, or mixed-type; it converges to MDI
    when features are orthogonal and high-SNR. This is the simplified faithful variant (not the full
    leave-one-out paper algorithm). See the module docstring for the full regime tag and
    appraisals/10_verdict.md.

    Parameters
    ----------
    x : pd.DataFrame
        The design matrix.
    y : pd.Series
        The binary target.
    random_state : int, default 0
        Forest seed.
    n_trees : int, default 60
        Number of trees used by the per-tree decomposition.
    classifier : RandomForestClassifier, optional
        A bootstrap RandomForest to use instead of the default (depth-6, 100-tree) forest.

    Returns
    -------
    pd.Series
        MDI+ importance per feature (indexed by column; higher is more important).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = pd.DataFrame(rng.standard_normal((400, 4)), columns=list("abcd"))
    >>> y = pd.Series((x["a"] + 0.3 * rng.standard_normal(400) > 0).astype(int))
    >>> imp = mdi_plus_importance(x, y)
    >>> imp.idxmax() == "a"
    True
    """
    forest = classifier if classifier is not None else _default_forest(random_state)
    forest.fit(x, y)
    xv = x.to_numpy(float)
    yv = y.to_numpy(float)
    n, n_features = xv.shape
    try:
        from sklearn.ensemble._forest import (
            _generate_sample_indices,
            _get_n_samples_bootstrap,
        )

        n_boot = _get_n_samples_bootstrap(n, forest.max_samples)
    except Exception:  # pragma: no cover
        _generate_sample_indices = None
        n_boot = n
    accumulated = np.zeros(n_features)
    used = 0
    for estimator in forest.estimators_[:n_trees]:
        tree = estimator.tree_
        internal = np.where(tree.feature >= 0)[0]
        if internal.size == 0:
            continue
        basis = (xv[:, tree.feature[internal]] <= tree.threshold[internal]).astype(
            float
        )
        basis = basis - basis.mean(axis=0, keepdims=True)
        groups = tree.feature[internal]
        if _generate_sample_indices is not None:
            in_bag = np.unique(
                _generate_sample_indices(estimator.random_state, n, n_boot)
            )
        else:  # pragma: no cover
            in_bag = np.arange(n)
        out_of_bag = np.setdiff1d(np.arange(n), in_bag, assume_unique=False)
        if out_of_bag.size < 5:
            continue
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(basis[in_bag], yv[in_bag])
        coefficients = ridge.coef_
        for feature in np.unique(groups):
            mask = groups == feature
            partial = basis[out_of_bag][:, mask] @ coefficients[mask]
            accumulated[feature] += partial.var()
        used += 1
    accumulated = accumulated / max(used, 1)
    return pd.Series(accumulated, index=x.columns)


def _per_sample_log_loss(
    y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    eps = 1e-15
    proba = np.clip(proba, eps, 1 - eps)
    class_to_column = {c: k for k, c in enumerate(classes)}
    columns = np.array([class_to_column[v] for v in y_true])
    chosen = proba[np.arange(len(y_true)), columns]
    return -np.log(chosen)


def conditional_predictive_impact(
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int = 0,
    n_splits: int = 4,
    n_estimators: int = 80,
    classifier: RandomForestClassifier | None = None,
) -> pd.DataFrame:
    r"""
    Conditional Predictive Impact (CPI; Watson and Wright 2021), with a calibrated significance test.

    For each cross-validation fold a classifier is fit on the training rows; on the test rows each
    feature is replaced by a conditional Gaussian knockoff (the feature regressed on the others by ridge
    plus resampled residual noise), and the per-sample increase in log loss is recorded. Pooled across
    folds, the mean increase is the CPI statistic and a one-sided paired t-test gives a p-value, so the
    method tests whether a feature matters - the calibrated test MDA lacks.

    Prefer CPI when you need a statistically valid test of whether a feature matters; it holds nominal
    size with good power (mildly elevated on the hardest high-correlation x low-SNR corner). See the
    module docstring for the full regime tag and appraisals/10_verdict.md.

    Parameters
    ----------
    x : pd.DataFrame
        The design matrix.
    y : pd.Series
        The target.
    random_state : int, default 0
        Seed for the folds, forest, and knockoff noise.
    n_splits : int, default 4
        Cross-validation folds.
    n_estimators : int, default 80
        Trees in the per-fold forest (used when ``classifier`` is None).
    classifier : RandomForestClassifier, optional
        A classifier template cloned per fold (defaults to a depth-6 forest).

    Returns
    -------
    pd.DataFrame
        Indexed by feature with columns ``importance`` (mean log-loss increase; higher = more important)
        and ``p_value`` (one-sided paired t-test of importance > 0).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> x = pd.DataFrame(rng.standard_normal((300, 3)), columns=list("abc"))
    >>> y = pd.Series((x["a"] + 0.3 * rng.standard_normal(300) > 0).astype(int))
    >>> cpi = conditional_predictive_impact(x, y)
    >>> bool(cpi.loc["a", "p_value"] < cpi.loc["c", "p_value"])
    True
    """
    from sklearn.base import clone

    columns = list(x.columns)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    xv = x.to_numpy(float)
    yv = y.to_numpy()
    differences: dict[str, list] = {c: [] for c in columns}
    template = (
        classifier
        if classifier is not None
        else _default_forest(random_state, n_estimators)
    )
    for i, (train, test) in enumerate(folds.split(x)):
        model = clone(template).fit(x.iloc[train], y.iloc[train])
        classes = model.classes_
        loss_true = _per_sample_log_loss(
            yv[test], model.predict_proba(x.iloc[test]), classes
        )
        rng = np.random.default_rng(random_state + 100 + i)
        for j, c in enumerate(columns):
            others = [k for k in range(len(columns)) if k != j]
            ridge = Ridge(alpha=1.0).fit(xv[train][:, others], xv[train][:, j])
            residual = xv[train][:, j] - ridge.predict(xv[train][:, others])
            sigma = residual.std() + 1e-9
            knockoff = ridge.predict(xv[test][:, others]) + sigma * rng.standard_normal(
                len(test)
            )
            x_knockoff = xv[test].copy()
            x_knockoff[:, j] = knockoff
            loss_knockoff = _per_sample_log_loss(
                yv[test],
                model.predict_proba(pd.DataFrame(x_knockoff, columns=columns)),
                classes,
            )
            differences[c].append(loss_knockoff - loss_true)
    importance, p_value = {}, {}
    for c in columns:
        d = np.concatenate(differences[c])
        importance[c] = float(d.mean())
        if d.std() < 1e-12:
            p_value[c] = 1.0
        else:
            t_stat, p_two = ttest_1samp(d, 0.0)
            p_value[c] = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)
    return pd.DataFrame(
        {"importance": pd.Series(importance), "p_value": pd.Series(p_value)}
    ).reindex(columns)
