"""
Concrete implementations of Imbalance Bars:
- ExpectedImbalanceBars
- FixedImbalanceBars
"""

from typing import Tuple, Union, Optional
import numpy as np

from RiskLabAI.data.structures.abstract_imbalance_bars import AbstractImbalanceBars
from RiskLabAI.utils.constants import *

# Assuming ewma is in utils
try:
    from RiskLabAI.utils.ewma import ewma 
except ImportError:
    # Fallback if ewma is not in utils (as seen in older files)
    def ewma(array: np.ndarray, window: int) -> np.ndarray:
        """Placeholder EWMA function."""
        if array.size == 0:
            return np.array([np.nan])
        return pd.Series(array).ewm(span=window).mean().values

class ExpectedImbalanceBars(AbstractImbalanceBars):
    """
    Concrete class for Imbalance Bars with a dynamic, EWMA-based
    Expected Number of Ticks (E[T]).
    """

    def __init__(
        self,
        bar_type: str,
        window_size_for_expected_n_ticks_estimation: int = 10000,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        expected_ticks_number_bounds: Optional[Tuple[float, float]] = None,
        analyse_thresholds: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        bar_type : str
            e.g., 'dollar_imbalance', 'tick_imbalance'.
        window_size_for_expected_n_ticks_estimation : int
            Window size for EWMA of E[T].
        initial_estimate_of_expected_n_ticks_in_bar : int
            Initial guess for E[T].
        window_size_for_expected_imbalance_estimation : int
            Window size for EWMA of E[b].
        expected_ticks_number_bounds : Tuple[float, float], optional
            (Lower, Upper) bounds to clamp E[T].
        analyse_thresholds : bool, default=False
            If True, store threshold data for analysis.
        """
        super().__init__(
            bar_type,
            window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds,
        )

        if expected_ticks_number_bounds is None:
            self.expected_ticks_number_lower_bound = 0.0
            self.expected_ticks_number_upper_bound = np.inf
        else:
            self.expected_ticks_number_lower_bound = (
                expected_ticks_number_bounds[0]
            )
            self.expected_ticks_number_upper_bound = (
                expected_ticks_number_bounds[1]
            )

    def _expected_number_of_ticks(self) -> float:
        """
        Calculate E[T] using an EWMA of previous bar tick counts.
        """
        prev_ticks_list = self.imbalance_bars_statistics[
            PREVIOUS_BARS_N_TICKS_LIST
        ]
        
        window = self.window_size_for_expected_n_ticks_estimation
        if window is None or window <= 0:
             # Fallback to simple mean if window is invalid
             return np.mean(prev_ticks_list)
             
        if not prev_ticks_list:
             # No bars yet, return initial estimate
             return self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]

        ewma_ticks = ewma(
            np.array(prev_ticks_list[-window:], dtype=float), window=window
        )[-1]

        return min(
            max(ewma_ticks, self.expected_ticks_number_lower_bound),
            self.expected_ticks_number_upper_bound,
        )


class FixedImbalanceBars(AbstractImbalanceBars):
    """
    Concrete class for Imbalance Bars with a fixed (constant)
    Expected Number of Ticks (E[T]).
    """
    def __init__(
        self,
        bar_type: str,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        analyse_thresholds: bool = False,
        window_size_for_expected_n_ticks_estimation: Optional[int] = None, 
    ):
        """
        Constructor.

        Parameters
        ----------
        bar_type : str
            e.g., 'dollar_imbalance', 'tick_imbalance'.
        initial_estimate_of_expected_n_ticks_in_bar : int
            The *fixed* value for E[T].
        window_size_for_expected_imbalance_estimation : int
            Window size for EWMA of E[b].
        analyse_thresholds : bool, default=False
            If True, store threshold data for analysis.
        window_size_for_expected_n_ticks_estimation : int, optional
            Ignored. Kept for consistent interface with factory.
        """
        super().__init__(
            bar_type,
            window_size_for_expected_n_ticks_estimation=None,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds=analyse_thresholds,
        )

    def _expected_number_of_ticks(self) -> float:
        """
        Return the fixed E[T] value.
        """
        return self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]