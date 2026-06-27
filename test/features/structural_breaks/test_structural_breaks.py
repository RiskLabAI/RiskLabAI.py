"""
Tests for features/structural_breaks/structural_breaks.py
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from RiskLabAI.features.structural_breaks.structural_breaks import (
    compute_beta,
    get_bsadf_sequence,
    get_bsadf_statistic,
    get_bubble_episodes,
    get_expanding_window_adf,
    get_gsadf_statistic,
    get_sadf_sequence,
    lag_dataframe,
    prepare_data,
    psy_minimum_window,
    simulate_psy_critical_values,
)


@pytest.fixture
def sample_series():
    """A simple series for testing."""
    return pd.DataFrame({"price": [1.0, 1.2, 1.1, 1.3, 1.5, 1.4]})


@pytest.fixture
def random_walk_series():
    """A non-stationary random walk."""
    rng = np.random.default_rng(42)
    log_price = np.log(100 + rng.normal(0, 1, 100).cumsum())
    return pd.DataFrame({"log_price": log_price})


def test_lag_dataframe(sample_series):
    """Test the lag_dataframe function."""
    lags = 2
    df = lag_dataframe(sample_series, lags)

    # Should create columns for lags 0, 1, 2
    assert "price_0" in df.columns
    assert "price_1" in df.columns
    assert "price_2" in df.columns

    # Check values
    assert np.isclose(df["price_0"].iloc[2], 1.1)
    assert np.isclose(df["price_1"].iloc[2], 1.2)
    assert np.isclose(df["price_2"].iloc[2], 1.0)
    assert pd.isna(df["price_2"].iloc[1])


def test_prepare_data(sample_series):
    """Test the prepare_data function."""
    lags = 1
    # series = [1.0, 1.2, 1.1, 1.3, 1.5, 1.4]
    # diff = [nan, 0.2, -0.1, 0.2, 0.2, -0.1]
    # x = lag_dataframe(diff, 1)
    #   diff_0 = [nan, 0.2, -0.1, 0.2, 0.2, -0.1]
    #   diff_1 = [nan, nan, 0.2, -0.1, 0.2, 0.2]
    # x (dropna) = (index 2-5)
    #   [ -0.1, 0.2 ]
    #   [ 0.2, -0.1 ]
    #   [ 0.2, 0.2 ]
    #   [ -0.1, 0.2 ]
    # x (replace col 0 with level)
    #   [ 1.1, 0.2 ]
    #   [ 1.3, -0.1 ]
    #   [ 1.5, 0.2 ]
    #   [ 1.4, 0.2 ]
    # y = diff (index 2-5)
    #   [ -0.1, 0.2, 0.2, -0.1 ]

    # FIX 1: Pass the Series, not the DataFrame
    y, x = prepare_data(sample_series["price"], constant="c", lags=lags)

    assert y.shape == (4, 1)
    # The expected x calculation in your test file comments was slightly off.
    # The lagged level (x_df) should be [nan, 1.0, 1.2, 1.1, 1.3, 1.5]
    # The lagged diff (lagged_deltas) is [nan, nan, 0.2, -0.1, 0.2, 0.2]
    # After join and dropna (from index 2):
    #   y      x = [level_l1, delta_l1, constant]
    # -0.1     [1.2,  0.2, 1.0]
    #  0.2     [1.1, -0.1, 1.0]
    #  0.2     [1.3,  0.2, 1.0]
    # -0.1     [1.5,  0.2, 1.0]
    assert x.shape == (4, 3)  # level, lag 1 diff, constant

    expected_y = np.array([[-0.1], [0.2], [0.2], [-0.1]])
    expected_x = np.array(
        [[1.2, 0.2, 1.0], [1.1, -0.1, 1.0], [1.3, 0.2, 1.0], [1.5, 0.2, 1.0]]
    )

    assert np.allclose(y, expected_y)
    assert np.allclose(x, expected_x)


def test_compute_beta_bugfix():
    """
    Test compute_beta against statsmodels to verify the bugfix.
    """
    # 1. Prepare data
    y_vec = np.array([1, 2, 3, 4, 5], dtype=float)
    x_vec = np.array([1.1, 1.9, 3.0, 4.1, 4.9], dtype=float)
    x_mat = sm.add_constant(x_vec)  # [const, x1]
    y_vec = y_vec.reshape(-1, 1)

    # 2. Get correct result from statsmodels
    model = sm.OLS(y_vec, x_mat).fit()
    sm_betas = model.params.reshape(-1, 1)
    sm_vcov = model.cov_params()

    # 3. Get result from our function
    my_betas, my_vcov = compute_beta(y_vec, x_mat)

    # 4. Compare
    assert np.allclose(my_betas, sm_betas)
    assert np.allclose(my_vcov, sm_vcov)


def test_adf_function(random_walk_series):
    """Test the main ADF loop."""
    # FIX 2: Pass the Series, not the DataFrame
    results = get_bsadf_statistic(
        log_price=random_walk_series["log_price"],
        min_sample_length=20,
        constant="c",
        lags=1,
    )

    assert "Time" in results

    assert "bsadf" in results
    assert isinstance(results["Time"], int)
    assert isinstance(results["bsadf"], float)
    assert np.isfinite(results["bsadf"])


# ---------------------------------------------------------------------------------
# GSADF / BSADF (Phillips-Shi-Yu 2015)
# ---------------------------------------------------------------------------------


@pytest.fixture
def random_walk():
    """A plain random walk (no bubble), as a level series."""
    rng = np.random.default_rng(7)
    return pd.Series(np.cumsum(rng.standard_normal(200)))


@pytest.fixture
def single_bubble():
    """A random walk with one embedded mildly-explosive episode over t=120..150."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(200))
    for t in range(120, 150):
        y[t] = 1.06 * y[t - 1] + rng.standard_normal()
    return pd.Series(y)


def test_psy_minimum_window():
    """PSY rule r0 = 0.01 + 1.8/sqrt(T), rounded, at least 3."""
    assert psy_minimum_window(200) == round((0.01 + 1.8 / 200**0.5) * 200)
    assert psy_minimum_window(10) >= 3


def test_bsadf_sequence_matches_get_bsadf_statistic(random_walk):
    """
    Replication: the fast BSADF sequence agrees with the validated per-endpoint
    ``get_bsadf_statistic`` on shared windows (the appraisal saw ~1e-14).
    """
    nmin = psy_minimum_window(len(random_walk))
    seq = get_bsadf_sequence(random_walk, nmin)
    for r2 in (nmin, 60, 120, 199):
        ref = get_bsadf_statistic(random_walk.iloc[: r2 + 1], nmin, "c", 0)["bsadf"]
        assert np.isclose(seq.loc[r2], ref, atol=1e-9, rtol=0)


def test_sadf_sequence_matches_expanding_window(random_walk):
    """The SADF date-stamp sequence equals the origin-anchored expanding-window ADF."""
    nmin = psy_minimum_window(len(random_walk))
    seq = get_sadf_sequence(random_walk, nmin)
    ref = get_expanding_window_adf(random_walk, nmin, "c", 0)
    assert np.allclose(seq.values, ref.values, atol=1e-9, rtol=0)


def test_gsadf_is_supremum_of_bsadf_sequence(single_bubble):
    """GSADF is the supremum of the BSADF sequence over all flexible windows."""
    nmin = psy_minimum_window(len(single_bubble))
    seq = get_bsadf_sequence(single_bubble, nmin)
    assert np.isclose(
        get_gsadf_statistic(single_bubble, nmin), np.nanmax(seq.values), atol=1e-12
    )


def test_critical_values_reproduce_psy_range():
    """
    Replication: simulated finite-sample GSADF 95% critical values fall in the
    Phillips-Shi-Yu range (~1.9-2.2 for the intercept-only spec) and exceed the
    single-window SADF critical value; the simulator is cached and deterministic.
    """
    prev = None
    for t_len in (100, 200, 400):
        cv = simulate_psy_critical_values(t_len, n_simulations=600, seed=123)
        assert 1.7 <= cv["gsadf_global_cv"] <= 2.5
        assert cv["gsadf_global_cv"] > cv["sadf_global_cv"]
        # GSADF critical value grows weakly with the sample length.
        if prev is not None:
            assert cv["gsadf_global_cv"] >= prev - 0.1
        prev = cv["gsadf_global_cv"]
    # Deterministic + cached: a second identical call returns the same object.
    a = simulate_psy_critical_values(100, n_simulations=600, seed=123)
    b = simulate_psy_critical_values(100, n_simulations=600, seed=123)
    assert a is b


def test_single_bubble_detected_and_dated(single_bubble):
    """A single embedded explosive episode is detected and roughly date-stamped."""
    nmin = psy_minimum_window(len(single_bubble))
    cv = simulate_psy_critical_values(
        len(single_bubble), nmin, n_simulations=300, seed=1
    )
    gsadf = get_gsadf_statistic(single_bubble, nmin)
    assert gsadf > cv["gsadf_global_cv"]  # bubble detected

    bsadf_seq = get_bsadf_sequence(single_bubble, nmin)
    episodes = get_bubble_episodes(bsadf_seq, cv["bsadf_sequence_cv"], min_duration=3)
    assert episodes  # at least one episode found
    # An episode brackets the true explosive window (t=120..150), within a tolerance.
    assert any(100 <= origination <= 140 for origination, _collapse in episodes)


def test_no_bubble_controls_size(random_walk):
    """Under a pure random walk no episode is flagged at the nominal level."""
    nmin = psy_minimum_window(len(random_walk))
    cv = simulate_psy_critical_values(len(random_walk), nmin, n_simulations=400, seed=5)
    bsadf_seq = get_bsadf_sequence(random_walk, nmin)
    episodes = get_bubble_episodes(
        bsadf_seq, cv["bsadf_sequence_cv"], min_duration=nmin
    )
    assert episodes == []


def test_short_series_returns_empty(random_walk):
    """A series shorter than the minimum window yields an empty sequence and NaN GSADF."""
    short = random_walk.iloc[:5]
    nmin = psy_minimum_window(200)  # far larger than the series
    seq = get_bsadf_sequence(short, nmin)
    assert len(seq) == 0
    assert np.isnan(get_gsadf_statistic(short, nmin))


def test_nan_values_are_filtered():
    """NaN values (e.g. FRED holiday gaps) are dropped before the test is run."""
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(120))
    series = pd.Series(y)
    series.iloc[10] = np.nan
    series.iloc[50] = np.nan
    nmin = psy_minimum_window(series.dropna().shape[0])
    gsadf = get_gsadf_statistic(series, nmin)
    assert np.isfinite(gsadf)
    # Equals the result on the explicitly cleaned series.
    assert np.isclose(gsadf, get_gsadf_statistic(series.dropna(), nmin))


def test_get_bubble_episodes_scalar_critical_value(single_bubble):
    """A scalar critical value is broadcast across the sequence."""
    nmin = psy_minimum_window(len(single_bubble))
    bsadf_seq = get_bsadf_sequence(single_bubble, nmin)
    episodes = get_bubble_episodes(bsadf_seq, 2.0, min_duration=3)
    assert episodes
    for origination, collapse in episodes:
        assert origination <= collapse
