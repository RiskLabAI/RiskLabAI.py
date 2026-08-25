r"""
Hardened (anti-degeneracy) imbalance and run bars.

de Prado's information-driven bars (AFML ch.2) sample a new bar when the cumulative signed imbalance
(or run) exceeds the product of the expected number of ticks E[T] and the expected absolute imbalance.
With an EWMA estimate of E[T] and the imbalance, the naive construction degenerates under some
parameterizations: the bar size can collapse toward a single tick (when the running imbalance estimate
makes the threshold tiny) or diverge toward extremely long bars (when E[T] runs away), so the bar series
is no longer usable. The hardened variant adds two guards that bound the bar size without touching the
well-behaved regime.

Admitted in Appraisal 19 (CONTRIBUTIONS_LEDGER 2026-06-27) as a correctness / robustness fix. Scope
note, verbatim from the verdict:

    replace the naive imbalance/run-bar construction with the guarded version - it removes the
    documented degeneracy (1-tick collapse / divergence) while leaving well-behaved parameterizations
    unchanged. This is a robustness admission, not a performance claim.

The guards are: (1) never close a bar before ``min_ticks`` ticks (prevents the 1-tick collapse), and
(2) force-close a bar at ``max_ticks`` ticks (prevents divergence); E[T] is additionally clamped to
``[min_ticks, max_ticks]`` via the base class's ``expected_ticks_number_bounds``. The naive
``ExpectedImbalanceBars`` / ``ExpectedRunBars`` are left unchanged and remain available; the hardened
classes only subclass them. Held-out confirmed on the sealed tick segment (appraisals/19_results,
HELDOUT.md): naive imbalance collapses (86.9% of bars <= 2 ticks, min 1) while hardened holds (0%
<= 2-tick, min 20); naive run diverges (11 bars) while hardened stays bounded; a well-behaved run
configuration is unchanged (6.2% bar-count difference). Evidence and caveats: appraisals/19_verdict.md.

References
----------
Lopez de Prado, M. (2018) Advances in Financial Machine Learning, ch. 2 (information-driven bars).
"""

from __future__ import annotations

from RiskLabAI.utils.constants import CUMULATIVE_TICKS

from .imbalance_bars import ExpectedImbalanceBars
from .run_bars import ExpectedRunBars


class HardenedExpectedImbalanceBars(ExpectedImbalanceBars):
    r"""
    Expected-imbalance bars with anti-degeneracy guards (Appraisal 19 robustness fix).

    Subclasses :class:`ExpectedImbalanceBars` and overrides only the bar-construction condition: a bar
    cannot close before ``min_ticks`` ticks (removes the 1-tick collapse) and is force-closed at
    ``max_ticks`` ticks (removes the divergence); E[T] is clamped to ``[min_ticks, max_ticks]``. Between
    the guards the original de Prado condition is used unchanged, so a well-parameterized construction is
    left essentially as-is. See the module docstring for the verbatim scope note and
    appraisals/19_verdict.md.

    Parameters
    ----------
    bar_type : str
        e.g. ``'dollar_imbalance'`` / ``'tick_imbalance'`` (passed through to the base class).
    min_ticks : int
        Minimum number of ticks before a bar may close (the collapse guard).
    max_ticks : int
        Maximum number of ticks before a bar is force-closed (the divergence guard).
    **kwargs
        Forwarded to :class:`ExpectedImbalanceBars` (``window_size_for_expected_n_ticks_estimation``,
        ``initial_estimate_of_expected_n_ticks_in_bar``, ``window_size_for_expected_imbalance_estimation``,
        ``analyse_thresholds``). ``expected_ticks_number_bounds`` is set to ``(min_ticks, max_ticks)``.

    Examples
    --------
    >>> bars = HardenedExpectedImbalanceBars(
    ...     bar_type="dollar_imbalance",
    ...     initial_estimate_of_expected_n_ticks_in_bar=200,
    ...     min_ticks=20,
    ...     max_ticks=5000,
    ... )
    >>> data = [("2020-01-01 10:00:00", 100.0, 10.0)] * 60
    >>> out = bars.construct_bars_from_data(data)   # no 1-tick collapse
    >>> isinstance(out, list)
    True
    """

    def __init__(self, *args, min_ticks: int, max_ticks: int, **kwargs):
        if min_ticks <= 0 or max_ticks < min_ticks:
            raise ValueError("require 0 < min_ticks <= max_ticks")
        kwargs["expected_ticks_number_bounds"] = (min_ticks, max_ticks)
        super().__init__(*args, **kwargs)
        self.min_ticks = min_ticks
        self.max_ticks = max_ticks

    def _bar_construction_condition(self, threshold: float) -> bool:
        n_ticks = self.base_statistics[CUMULATIVE_TICKS]
        if n_ticks < self.min_ticks:
            return False
        if n_ticks >= self.max_ticks:
            return True
        return super()._bar_construction_condition(threshold)


class HardenedExpectedRunBars(ExpectedRunBars):
    r"""
    Expected-run bars with anti-degeneracy guards (Appraisal 19 robustness fix).

    Subclasses :class:`ExpectedRunBars` and overrides only the bar-construction condition with the same
    two guards as :class:`HardenedExpectedImbalanceBars`: no close before ``min_ticks`` ticks, force-close
    at ``max_ticks`` ticks, E[T] clamped to ``[min_ticks, max_ticks]``. See the module docstring for the
    verbatim scope note and appraisals/19_verdict.md.

    Parameters
    ----------
    bar_type : str
        e.g. ``'dollar_run'`` / ``'tick_run'``.
    min_ticks : int
        Minimum ticks before a bar may close (collapse guard).
    max_ticks : int
        Maximum ticks before a bar is force-closed (divergence guard).
    **kwargs
        Forwarded to :class:`ExpectedRunBars`; ``expected_ticks_number_bounds`` is set to
        ``(min_ticks, max_ticks)``.

    Examples
    --------
    >>> bars = HardenedExpectedRunBars(
    ...     bar_type="dollar_run",
    ...     window_size_for_expected_n_ticks_estimation=200,
    ...     initial_estimate_of_expected_n_ticks_in_bar=200,
    ...     window_size_for_expected_imbalance_estimation=10000,
    ...     min_ticks=20,
    ...     max_ticks=5000,
    ... )
    >>> data = [("2020-01-01 10:00:00", 100.0, 10.0)] * 60
    >>> out = bars.construct_bars_from_data(data)
    >>> isinstance(out, list)
    True
    """

    def __init__(self, *args, min_ticks: int, max_ticks: int, **kwargs):
        if min_ticks <= 0 or max_ticks < min_ticks:
            raise ValueError("require 0 < min_ticks <= max_ticks")
        kwargs["expected_ticks_number_bounds"] = (min_ticks, max_ticks)
        super().__init__(*args, **kwargs)
        self.min_ticks = min_ticks
        self.max_ticks = max_ticks

    def _bar_construction_condition(self, threshold: float) -> bool:
        n_ticks = self.base_statistics[CUMULATIVE_TICKS]
        if n_ticks < self.min_ticks:
            return False
        if n_ticks >= self.max_ticks:
            return True
        return super()._bar_construction_condition(threshold)
