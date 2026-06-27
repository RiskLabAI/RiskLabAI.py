"""
Implements the core logic for the (G)SADF (Generalized) Supreme
Augmented Dickey-Fuller test for structural breaks.

References:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 17. (SADF, Phillips-Wu-Yu 2011.)
    Phillips, P. C. B., Shi, S. and Yu, J. (2015) Testing for multiple bubbles:
    historical episodes of exuberance and collapse in the S&P 500.
    International Economic Review, 56(4), 1043-1078. (GSADF / BSADF.)
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def lag_dataframe(
    market_data: pd.DataFrame, lags: Union[int, list[int]]
) -> pd.DataFrame:
    """
    Apply lags to a DataFrame.

    Reference:
        Snippet 17.3, Page 253.

    Parameters
    ----------
    market_data : pd.DataFrame
        DataFrame of price or log price.
    lags : Union[int, List[int]]
        An integer number of lags (e.g., 3 creates lags 0, 1, 2, 3)
        or a specific list of lags to create.

    Returns
    -------
    pd.DataFrame
        A DataFrame with lagged columns, e.g., 'price_0', 'price_1', ...
    """
    lagged_parts = []

    if isinstance(lags, int):
        lags_list = range(lags + 1)
    else:
        lags_list = [int(lag) for lag in lags]

    for lag in lags_list:
        lagged_data = market_data.shift(lag)
        lagged_data.columns = [f"{col}_{lag}" for col in market_data.columns]
        lagged_parts.append(lagged_data)

    return pd.concat(lagged_parts, axis=1)


def prepare_data(
    log_price_series: pd.Series, constant: str, lags: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the y and X matrices for ADF regression.

    Reference:
        Snippet 17.3, Page 253.

    Parameters
    ----------
    log_price_series : pd.Series  # <-- CHANGED: Accept Series
        Series of log prices.
    constant : str
        Type of regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int
        Number of lags to include.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - y_df: The dependent variable (delta log price).
        - x_df: The independent variables (lagged level, lagged deltas, constants).
    """
    # <-- ADDED: Convert univariate series to frame for internal processing
    log_price = log_price_series.to_frame()

    price_diff = log_price.diff().dropna()
    y_df = price_diff

    # 1. Lagged level
    x_df = log_price.shift(1).copy()
    x_df.columns = ["level_l1"]

    # 2. Lagged deltas
    if lags > 0:
        lagged_deltas = price_diff.shift(1)
        lagged_deltas.columns = ["delta_l1"]

        if lags > 1:
            for i in range(2, lags + 1):
                lagged_deltas[f"delta_l{i}"] = price_diff.shift(i)

        x_df = x_df.join(lagged_deltas, how="outer")

    # 3. Add constants
    if constant == "c":
        x_df["constant"] = 1
    elif constant == "ct":
        x_df["constant"] = 1
        x_df["trend"] = np.arange(1, len(x_df) + 1)
    elif constant == "ctt":
        x_df["constant"] = 1
        x_df["trend"] = np.arange(1, len(x_df) + 1)
        x_df["trend_sq"] = x_df["trend"] ** 2

    # Align y and X by dropping NaNs created by lagging
    combined = y_df.join(x_df, how="inner").dropna()

    y_df = combined.iloc[:, [0]]
    x_df = combined.iloc[:, 1:]

    return y_df, x_df


def compute_beta(
    y_window: np.ndarray, x_window: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute OLS beta coefficients and their variance.

    Reference:
        Snippet 17.2, Page 251.

    Parameters
    ----------
    y_window : np.ndarray
        Window of the dependent variable.
    x_window : np.ndarray
        Window of the independent variables.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - beta_mean: The OLS coefficients.
        - beta_variance: The variance-covariance matrix of the coefficients.
    """
    try:
        xt_x_inv = np.linalg.inv(x_window.T @ x_window)
        xt_y = x_window.T @ y_window

        beta_mean = xt_x_inv @ xt_y

        error = y_window - (x_window @ beta_mean)
        variance_e = (error.T @ error) / (x_window.shape[0] - x_window.shape[1])
        beta_variance = variance_e * xt_x_inv

        return beta_mean, beta_variance

    except np.linalg.LinAlgError:
        # Handle singular matrix
        return np.full((x_window.shape[1], 1), np.nan), np.full(
            (x_window.shape[1], x_window.shape[1]), np.nan
        )


def get_expanding_window_adf(
    log_price: pd.Series,
    min_sample_length: int,
    constant: str,
    lags: int,
) -> pd.Series:
    """
    Compute the ADF t-statistic over an expanding window.

    This is useful for plotting the evolution of the test statistic over time.

    Reference:
        Based on Snippet 17.2, Page 251.

    Parameters
    ----------
    log_price : pd.Series
        Series of log prices.
    min_sample_length : int
        The minimum number of samples to start the expanding window.
    constant : str
        Type of regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int
        Number of lags to include in the regression.

    Returns
    -------
    pd.Series
        A Series of ADF t-statistics, indexed by timestamp.
    """
    # <-- CHANGED: Pass Series directly to prepare_data
    y_df, x_df = prepare_data(log_price, constant=constant, lags=lags)

    adf_stats = []
    timestamps = []

    for i in range(min_sample_length, y_df.shape[0] + 1):
        y_window = y_df.iloc[:i].values
        x_window = x_df.iloc[:i].values

        beta_mean, beta_variance = compute_beta(y_window, x_window)

        if np.isnan(beta_variance[0, 0]):
            t_stat = np.nan
        else:
            beta_std_level = beta_variance[0, 0] ** 0.5
            if beta_std_level == 0:
                t_stat = -np.inf if beta_mean[0, 0] < 0 else np.inf
            else:
                t_stat = beta_mean[0, 0] / beta_std_level

        adf_stats.append(t_stat)
        timestamps.append(y_df.index[i - 1])

    return pd.Series(adf_stats, index=timestamps)


def get_bsadf_statistic(
    log_price: pd.Series,  # <-- CHANGED: Accept Series
    min_sample_length: int,
    constant: str,
    lags: int,
) -> dict[str, Any]:
    """
    Compute the Backward Supremum ADF (BSADF) statistic.

    This test runs an expanding ADF test starting from every possible
    point in the series and finds the supremum (highest) t-statistic.
    This is used to detect the *origination* of a bubble.

    Reference:
        Snippet 17.4, Page 253. (Renamed from `adf` for clarity).

    Parameters
    ----------
    log_price : pd.Series  # <-- CHANGED
        Series of log prices.
    min_sample_length : int
        Minimum sample length for each ADF test.
    constant : str
        Type of regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int
        Number of lags to include.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'Time': The timestamp of the end of the series.
        - 'bsadf': The BSADF statistic (the supremum ADF).
    """
    # 1. Prepare the full X, y matrices
    # <-- CHANGED: Pass Series directly
    y, x = prepare_data(log_price, constant=constant, lags=lags)

    # 2. Define all possible start points
    # <-- BUG FIX: Removed '+ lags' from the range
    start_points = range(0, y.shape[0] - min_sample_length + 1)
    bsadf = -np.inf  # Supremum ADF

    y_np, x_np = y.values, x.values

    # 3. Loop over all expanding windows
    for start in start_points:
        y_window, x_window = y_np[start:], x_np[start:]

        # 4. Compute ADF regression for this window
        beta_mean, beta_variance = compute_beta(y_window, x_window)

        if np.isnan(beta_variance[0, 0]):
            continue

        # 5. Get t-statistic for the first coefficient (the level)
        beta_mean_level = beta_mean[0, 0]
        beta_std_level = beta_variance[0, 0] ** 0.5

        if beta_std_level == 0:
            t_stat = -np.inf if beta_mean_level < 0 else np.inf
        else:
            t_stat = beta_mean_level / beta_std_level

        if t_stat > bsadf:
            bsadf = t_stat

    return {"Time": log_price.index[-1], "bsadf": bsadf}


# ----------------------------------------------------------------------------------
# GSADF / BSADF (Phillips-Shi-Yu 2015): the generalized sup-ADF and its backward
# sequence, with date-stamping and a finite-sample critical-value simulator.
#
# de Prado (AFML ch.17) ships the single-window SADF: the forward-expanding ADF is
# anchored at the sample origin, so several explosive episodes dilute and mask one
# another and SADF collapses them to about one. GSADF varies *both* window endpoints
# and the backward BSADF sequence dates each origination and collapse, recovering and
# counting multiple bubbles where SADF sees one. The functions below build directly
# on the validated ``get_bsadf_statistic`` / ``get_expanding_window_adf`` above (and
# agree with them to ~1e-9 on shared windows; see the structural-breaks tests).
#
# Admitted in Appraisal 05 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tag, verbatim:
#   Prefer GSADF/BSADF over single-window SADF when a series may contain more than
#   one explosive episode (it recovers and counts each). For a single suspected
#   bubble, SADF is at least as good. Use seasonally-adjusted data and the simulated
#   finite-sample critical values; GSADF over-flags strongly trending or seasonal
#   series and is mildly oversized in large samples, so treat broad flags cautiously.
# Evidence and caveats: appraisals/05_verdict.md.
# ----------------------------------------------------------------------------------


def psy_minimum_window(sample_length: int) -> int:
    """
    Phillips-Shi-Yu minimum window length, in observations.

    PSY (2015) set the minimum window fraction to r0 = 0.01 + 1.8 / sqrt(T) and take
    the minimum window as round(r0 * T), the standard rule used to simulate the
    finite-sample critical values.

    Parameters
    ----------
    sample_length : int
        The series length T (number of price observations).

    Returns
    -------
    int
        The minimum window length in observations (at least 3).
    """
    r0 = 0.01 + 1.8 / (sample_length**0.5)
    return max(int(round(r0 * sample_length)), 3)


def _psy_sadf_bsadf_sequences(
    y: np.ndarray, min_sample_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast prefix-sum kernel for the PSY-standard ADF specification (intercept, no lag
    augmentation: Delta y_t = alpha + beta y_{t-1} + e_t).

    Returns ``(sadf_seq, bsadf_seq)``, each of length ``len(y)`` and indexed by the
    window endpoint (price index), with ``NaN`` before the first valid endpoint:

    - ``sadf_seq[r2]``  = ADF(0, r2)                     (forward-expanding, PWY)
    - ``bsadf_seq[r2]`` = sup_{r1 in [0, r2 - nmin]} ADF(r1, r2)   (backward sup, PSY)

    The ADF t-statistic of the lagged-level coefficient for any window is computed in
    O(1) from prefix sums of the regression observations, so the whole sequence is
    O(T^2). Clean-room from Phillips-Shi-Yu (2015); it agrees with the loop-based
    ``get_bsadf_statistic`` / ``get_expanding_window_adf`` to ~1e-9 on shared windows.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    length = y.shape[0]
    sadf = np.full(length, np.nan)
    bsadf = np.full(length, np.nan)
    if length < min_sample_length + 1:
        return sadf, bsadf

    # Regression observation i (0-based) uses level x_i = y[i] and delta d_i = y[i+1]-y[i].
    x = y[:-1]
    d = y[1:] - y[:-1]

    # Prefix sums (length m+1, leading 0) so any window's sufficient statistics are O(1).
    cx = np.concatenate(([0.0], np.cumsum(x)))
    cxx = np.concatenate(([0.0], np.cumsum(x * x)))
    cd = np.concatenate(([0.0], np.cumsum(d)))
    cxd = np.concatenate(([0.0], np.cumsum(x * d)))
    cdd = np.concatenate(([0.0], np.cumsum(d * d)))

    for r2 in range(min_sample_length, length):
        # Window [r1, r2] uses regression observations [r1, r2); n = r2 - r1.
        r1 = np.arange(0, r2 - min_sample_length + 1)
        n = (r2 - r1).astype(np.float64)
        sx = cx[r2] - cx[r1]
        sxx = cxx[r2] - cxx[r1]
        sd = cd[r2] - cd[r1]
        sxd = cxd[r2] - cxd[r1]
        sdd = cdd[r2] - cdd[r1]
        den = n * sxx - sx * sx
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (n * sxd - sx * sd) / den
            alpha = (sd - beta * sx) / n
            ssr = np.maximum(sdd - alpha * sd - beta * sxd, 0.0)
            var_beta = (ssr / (n - 2)) * n / den
            t_stat = beta / np.sqrt(var_beta)
        valid = (den > 0.0) & (n >= 3) & (var_beta > 0.0)
        t_stat = np.where(valid, t_stat, -np.inf)
        best = t_stat.max()
        bsadf[r2] = best if np.isfinite(best) else np.nan
        # r1 == 0 is the forward-expanding (origin-anchored) ADF = SADF date-stamp.
        sadf[r2] = t_stat[0] if valid[0] else np.nan

    return sadf, bsadf


def get_sadf_sequence(
    log_price: pd.Series,
    min_sample_length: int,
    constant: str = "c",
    lags: int = 0,
) -> pd.Series:
    """
    Forward-expanding ADF (SADF) sequence, indexed by window endpoint.

    This is the date-stamping sequence for de Prado's single-window SADF: each point
    is the ADF t-statistic of an expanding window anchored at the sample origin. It is
    a thin wrapper over :func:`get_expanding_window_adf` (and the fast prefix-sum
    kernel for the standard intercept-only, no-lag specification) so the SADF and GSADF
    date-stamps share an aligned index.

    Parameters
    ----------
    log_price : pd.Series
        Series of log prices. ``NaN`` values (e.g. FRED holiday gaps) are dropped.
    min_sample_length : int
        Minimum window length in observations (see :func:`psy_minimum_window`).
    constant : str, default "c"
        Regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int, default 0
        Number of lagged-difference terms (PSY simulate with 0).

    Returns
    -------
    pd.Series
        The SADF t-statistic at each endpoint, indexed by timestamp.
    """
    log_price = log_price.dropna()
    if constant == "c" and lags == 0:
        sadf, _ = _psy_sadf_bsadf_sequences(log_price.to_numpy(), min_sample_length)
        endpoints = log_price.index[min_sample_length:]
        return pd.Series(sadf[min_sample_length:], index=endpoints)
    return get_expanding_window_adf(log_price, min_sample_length, constant, lags)


def get_bsadf_sequence(
    log_price: pd.Series,
    min_sample_length: int,
    constant: str = "c",
    lags: int = 0,
) -> pd.Series:
    """
    Backward sup-ADF (BSADF) sequence, indexed by window endpoint (Phillips-Shi-Yu).

    For each endpoint the statistic is the supremum of the ADF t-statistic over all
    window start points, so a later episode is tested on its own sub-sample rather than
    diluted by the earlier sample. Comparing this sequence against a critical-value
    sequence (see :func:`simulate_psy_critical_values`) date-stamps each bubble in real
    time; see :func:`get_bubble_episodes`.

    For the standard intercept-only, no-lag specification this uses a fast O(T^2)
    prefix-sum kernel; for other constants or ``lags > 0`` it reuses the validated
    :func:`get_bsadf_statistic` on expanding prefixes. Both agree to ~1e-9.

    Parameters
    ----------
    log_price : pd.Series
        Series of log prices. ``NaN`` values are dropped.
    min_sample_length : int
        Minimum window length in observations (see :func:`psy_minimum_window`).
    constant : str, default "c"
        Regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int, default 0
        Number of lagged-difference terms.

    Returns
    -------
    pd.Series
        The BSADF t-statistic at each endpoint, indexed by timestamp.
    """
    log_price = log_price.dropna()
    if constant == "c" and lags == 0:
        _, bsadf = _psy_sadf_bsadf_sequences(log_price.to_numpy(), min_sample_length)
        endpoints = log_price.index[min_sample_length:]
        return pd.Series(bsadf[min_sample_length:], index=endpoints)

    values, timestamps = [], []
    first_end = min_sample_length + lags + 1
    for end in range(first_end, len(log_price) + 1):
        result = get_bsadf_statistic(
            log_price.iloc[:end], min_sample_length, constant, lags
        )
        values.append(result["bsadf"])
        timestamps.append(result["Time"])
    return pd.Series(values, index=timestamps)


def get_gsadf_statistic(
    log_price: pd.Series,
    min_sample_length: int,
    constant: str = "c",
    lags: int = 0,
) -> float:
    """
    Generalized sup-ADF (GSADF) statistic (Phillips-Shi-Yu 2015).

    The GSADF is the supremum of the backward sup-ADF (BSADF) sequence over all
    flexible windows, i.e. ``max`` of :func:`get_bsadf_sequence`. A series is flagged
    as containing at least one explosive episode when the GSADF exceeds its
    finite-sample critical value (:func:`simulate_psy_critical_values`).

    Prefer GSADF/BSADF over single-window SADF when a series may contain more than one
    explosive episode (it recovers and counts each); for a single suspected bubble,
    SADF is at least as good. Use seasonally-adjusted data and the simulated
    finite-sample critical values; GSADF over-flags strongly trending or seasonal
    series and is mildly oversized in large samples, so treat broad flags cautiously.
    (Admitted in Appraisal 05; see appraisals/05_verdict.md.)

    Parameters
    ----------
    log_price : pd.Series
        Series of log prices. ``NaN`` values are dropped.
    min_sample_length : int
        Minimum window length in observations (see :func:`psy_minimum_window`).
    constant : str, default "c"
        Regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int, default 0
        Number of lagged-difference terms.

    Returns
    -------
    float
        The GSADF statistic, or ``NaN`` if the series is too short.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(200))          # random walk
    >>> for t in range(120, 150):                        # embed an explosive episode
    ...     y[t] = 1.06 * y[t - 1] + rng.standard_normal()
    >>> price = pd.Series(y)
    >>> nmin = psy_minimum_window(len(price))
    >>> gsadf = get_gsadf_statistic(price, nmin)
    >>> cv = simulate_psy_critical_values(len(price), nmin, n_simulations=200)
    >>> bool(gsadf > cv["gsadf_global_cv"])              # bubble detected
    True
    """
    bsadf_seq = get_bsadf_sequence(log_price, min_sample_length, constant, lags)
    values = bsadf_seq.to_numpy()
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmax(values))


def get_bubble_episodes(
    statistic_sequence: pd.Series,
    critical_value_sequence: Union[pd.Series, np.ndarray, float],
    min_duration: int = 1,
) -> list[tuple[Any, Any]]:
    """
    Date-stamp explosive episodes from a sup-ADF sequence (Phillips-Shi-Yu 2015).

    An episode is a maximal run of endpoints where the statistic exceeds its
    critical value and that lasts at least ``min_duration`` periods. The origination
    is the first such period; the collapse is the first period the statistic falls
    back below the critical value (or the last endpoint if it is still above at the
    end of the sample). The same rule applied to the BSADF sequence date-stamps GSADF
    episodes and applied to the SADF sequence date-stamps the single-window detector,
    so the two are directly comparable.

    Parameters
    ----------
    statistic_sequence : pd.Series
        A BSADF or SADF sequence (see :func:`get_bsadf_sequence`,
        :func:`get_sadf_sequence`).
    critical_value_sequence : pd.Series, np.ndarray or float
        The per-endpoint critical-value sequence (aligned with
        ``statistic_sequence``) or a single scalar critical value.
    min_duration : int, default 1
        Minimum run length, in periods, for a run to count as an episode. PSY use a
        small minimum such as ``round(log(T))``.

    Returns
    -------
    list of (origination, collapse)
        Index labels (timestamps) bounding each detected episode, in order.
    """
    stat = statistic_sequence.to_numpy(dtype=float)
    index = statistic_sequence.index
    n = stat.shape[0]
    if isinstance(critical_value_sequence, pd.Series):
        cv = critical_value_sequence.reindex(index).to_numpy(dtype=float)
    elif np.isscalar(critical_value_sequence):
        cv = np.full(n, float(critical_value_sequence))
    else:
        cv = np.asarray(critical_value_sequence, dtype=float)

    above = np.isfinite(stat) & np.isfinite(cv) & (stat > cv)
    episodes: list[tuple[Any, Any]] = []
    t = 0
    while t < n:
        if above[t]:
            start = t
            while t < n and above[t]:
                t += 1
            end = t  # first endpoint back below the critical value (exclusive)
            if end - start >= min_duration:
                collapse = index[end] if end < n else index[-1]
                episodes.append((index[start], collapse))
        else:
            t += 1
    return episodes


_CRITICAL_VALUE_CACHE: dict[tuple, dict[str, Any]] = {}


def simulate_psy_critical_values(
    sample_length: int,
    min_sample_length: Optional[int] = None,
    constant: str = "c",
    lags: int = 0,
    n_simulations: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Finite-sample critical values for SADF and GSADF, by simulating the random-walk
    null (the standard Phillips-Shi-Yu approach), with caching.

    The ADF-family statistics are pivotal under the random-walk null, so the values
    depend only on the sample length, the minimum window and the regression
    specification, not on the data's level or scale. For each of ``n_simulations``
    independent random-walk paths the SADF and BSADF sequences are computed; the
    ``level`` quantile of the global statistic across paths gives the global critical
    value (used for detection), and the ``level`` quantile per endpoint gives the
    critical-value sequence (used for date-stamping; see :func:`get_bubble_episodes`).

    Parameters
    ----------
    sample_length : int
        The series length T to simulate.
    min_sample_length : int, optional
        Minimum window length in observations. Defaults to
        :func:`psy_minimum_window`.
    constant : str, default "c"
        Regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int, default 0
        Number of lagged-difference terms.
    n_simulations : int, default 2000
        Number of random-walk paths.
    level : float, default 0.95
        Quantile of the null distribution (a 95% test targets a 5% size).
    seed : int, default 0
        Base seed; path ``s`` uses ``seed + s`` for reproducibility.

    Returns
    -------
    dict
        Keys ``sample_length``, ``min_sample_length``, ``level``,
        ``sadf_global_cv``, ``gsadf_global_cv`` (floats), and
        ``sadf_sequence_cv``, ``bsadf_sequence_cv`` (per-endpoint critical-value
        arrays aligned with :func:`get_sadf_sequence` / :func:`get_bsadf_sequence`).
    """
    if min_sample_length is None:
        min_sample_length = psy_minimum_window(sample_length)

    cache_key = (
        sample_length,
        min_sample_length,
        constant,
        lags,
        n_simulations,
        level,
        seed,
    )
    if cache_key in _CRITICAL_VALUE_CACHE:
        return _CRITICAL_VALUE_CACHE[cache_key]

    fast = constant == "c" and lags == 0
    sadf_paths = np.full((n_simulations, sample_length), np.nan)
    bsadf_paths = np.full((n_simulations, sample_length), np.nan)
    for s in range(n_simulations):
        rng = np.random.default_rng(seed + s)
        y = np.cumsum(rng.standard_normal(sample_length))  # random-walk null
        if fast:
            sadf, bsadf = _psy_sadf_bsadf_sequences(y, min_sample_length)
        else:
            price = pd.Series(y)
            sadf = (
                get_sadf_sequence(price, min_sample_length, constant, lags)
                .reindex(range(sample_length))
                .to_numpy()
            )
            bsadf = (
                get_bsadf_sequence(price, min_sample_length, constant, lags)
                .reindex(range(sample_length))
                .to_numpy()
            )
        sadf_paths[s] = sadf
        bsadf_paths[s] = bsadf

    global_sadf = np.nanmax(sadf_paths, axis=1)
    global_gsadf = np.nanmax(bsadf_paths, axis=1)
    # Quantiles only over the valid endpoint columns (the leading columns are all NaN).
    sadf_seq_cv = np.nanquantile(sadf_paths[:, min_sample_length:], level, axis=0)
    bsadf_seq_cv = np.nanquantile(bsadf_paths[:, min_sample_length:], level, axis=0)

    result = {
        "sample_length": sample_length,
        "min_sample_length": min_sample_length,
        "level": level,
        "sadf_global_cv": float(np.nanquantile(global_sadf, level)),
        "gsadf_global_cv": float(np.nanquantile(global_gsadf, level)),
        "sadf_sequence_cv": sadf_seq_cv,
        "bsadf_sequence_cv": bsadf_seq_cv,
    }
    _CRITICAL_VALUE_CACHE[cache_key] = result
    return result
