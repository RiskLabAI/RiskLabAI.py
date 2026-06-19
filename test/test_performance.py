"""
Correctness tests for the vectorized hot-path rewrites (Phase 4, performance).

Each vectorized implementation is checked against an independent brute-force
reference that encodes the plain mathematical definition. This guarantees the
optimized versions return the same values as the straightforward loops they
replaced, across ordinary and edge cases (leading NaNs, NaT vertical barriers,
both trade sides, and no-touch paths).
"""

import numpy as np
import pandas as pd

from RiskLabAI.backtest.bet_sizing import mp_avg_active_signals
from RiskLabAI.data.differentiation.differentiation import (
    calculate_weights_std,
    fractional_difference_std,
)
from RiskLabAI.data.labeling.labeling import triple_barrier


# --------------------------------------------------------------------------- #
# fractional_difference_std  (expanding-window weighted sum -> convolution)
# --------------------------------------------------------------------------- #
def _frac_diff_std_reference(series, degree, threshold=0.01):
    weights = calculate_weights_std(degree, series.shape[0])
    weights_cumsum = np.cumsum(np.abs(weights))
    weights_cumsum /= weights_cumsum[-1]
    skip = np.searchsorted(weights_cumsum, threshold)
    result = pd.DataFrame(index=series.index, columns=series.columns, dtype=float)
    for name in series.columns:
        s = series[[name]].ffill().dropna()
        if s.empty or s.shape[0] < skip:
            continue
        arr = s.to_numpy()
        for iloc in range(skip, arr.shape[0]):
            result.loc[s.index[iloc], name] = np.dot(
                weights[-(iloc + 1) :].T, arr[: iloc + 1]
            )[0, 0]
    return result.dropna(how="all")


def test_fractional_difference_std_matches_reference():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=600, freq="min")
    values = np.cumsum(rng.standard_normal(600)) + 100
    series = pd.DataFrame(
        {"close": values, "other": values * 0.5 + rng.standard_normal(600)},
        index=idx,
    )
    series.iloc[:5, 0] = np.nan  # leading NaNs are dropped by ffill().dropna()

    fast = fractional_difference_std(series, 0.4)
    reference = _frac_diff_std_reference(series, 0.4)

    assert fast.shape == reference.shape
    assert fast.index.equals(reference.index)
    common = fast.index.intersection(reference.index)
    assert np.allclose(
        fast.loc[common].to_numpy(),
        reference.loc[common].to_numpy(),
        atol=1e-9,
        equal_nan=True,
    )


# --------------------------------------------------------------------------- #
# mpAvgActiveSignals  (interval stabbing -> prefix sums + searchsorted)
# --------------------------------------------------------------------------- #
def _avg_active_reference(signals, molecule):
    out = pd.Series(dtype=float)
    for loc in molecule:
        active = (signals.index.values <= loc) & (
            (loc < signals["t1"]) | pd.isnull(signals["t1"])
        )
        idx = signals[active].index
        out[loc] = signals.loc[idx, "signal"].mean() if len(idx) > 0 else 0.0
    return out


def test_mp_avg_active_signals_matches_reference():
    rng = np.random.default_rng(7)
    n = 500
    starts = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 50_000, n)), "s"
    )
    durations = pd.to_timedelta(rng.integers(1, 4000, n), "s")
    t1 = pd.Series(starts + durations)
    t1[rng.random(n) < 0.1] = pd.NaT  # some open-ended signals
    signals = pd.DataFrame(
        {"t1": t1.values, "signal": rng.standard_normal(n)}, index=starts
    )
    signals = signals[~signals.index.duplicated(keep="first")]

    time_points = sorted(set(signals["t1"].dropna().values).union(signals.index.values))

    fast = mp_avg_active_signals(signals, time_points)
    reference = _avg_active_reference(signals, time_points)

    assert np.allclose(fast.values, reference.values, atol=1e-12, equal_nan=True)


def test_mp_avg_active_signals_empty_molecule():
    signals = pd.DataFrame(
        {"t1": [pd.Timestamp("2020-01-02")], "signal": [1.0]},
        index=[pd.Timestamp("2020-01-01")],
    )
    assert len(mp_avg_active_signals(signals, [])) == 0


# --------------------------------------------------------------------------- #
# triple_barrier  (per-event pandas slicing -> positional numpy indexing)
# --------------------------------------------------------------------------- #
def _triple_barrier_reference(close, events, ptsl, molecule):
    ef = events.loc[molecule]
    output = pd.DataFrame(index=ef.index)
    output["End Time"] = ef["End Time"]
    pt = (
        ptsl[0] * ef["Base Width"] if ptsl[0] > 0 else pd.Series(np.inf, index=ef.index)
    )
    sl = (
        -ptsl[1] * ef["Base Width"]
        if ptsl[1] > 0
        else pd.Series(-np.inf, index=ef.index)
    )
    side = ef.get("Side", pd.Series(1.0, index=ef.index))
    for loc, vbt in ef["End Time"].fillna(close.index[-1]).items():
        path = close.loc[loc:vbt]
        returns = np.log(path / close[loc]) * side.at[loc]
        output.loc[loc, "stop_loss"] = returns[returns < sl.at[loc]].index.min()
        output.loc[loc, "profit_taking"] = returns[returns > pt.at[loc]].index.min()
    output["End Time"] = output.min(axis=1)
    return output.drop(columns=["stop_loss", "profit_taking"])


def test_triple_barrier_matches_reference_randomized():
    rng = np.random.default_rng(11)
    for _ in range(25):
        n = int(rng.integers(60, 300))
        idx = pd.date_range("2020-01-01", periods=n, freq="min")
        close = pd.Series(
            np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100, index=idx
        )

        positions = np.sort(
            rng.choice(n - 2, size=int(rng.integers(3, 25)), replace=False)
        )
        vertical = []
        for p in positions:
            if rng.random() < 0.25:
                vertical.append(pd.NaT)  # open vertical barrier
            else:
                vertical.append(idx[min(p + int(rng.integers(1, 30)), n - 1)])
        events = pd.DataFrame(
            {
                "End Time": pd.to_datetime(vertical),
                "Base Width": rng.uniform(0.005, 0.05, size=len(positions)),
                "Side": rng.choice([1.0, -1.0], size=len(positions)),
            },
            index=idx[positions],
        )
        ptsl = [float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.5, 2.0))]
        molecule = list(events.index)

        fast = triple_barrier(close, events, ptsl, molecule)
        reference = _triple_barrier_reference(close, events, ptsl, molecule)
        # check_dtype=False: the timestamps are identical, but pandas >= 3 can
        # land the two construction paths on different datetime *resolutions*
        # (ns vs us). We assert the values match, not the resolution unit.
        pd.testing.assert_series_equal(
            fast["End Time"],
            reference["End Time"],
            check_names=False,
            check_dtype=False,
        )
