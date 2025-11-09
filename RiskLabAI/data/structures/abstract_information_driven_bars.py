"""
Abstract base class for Information-Driven Bars (Imbalance and Run bars).
"""

from abc import abstractmethod
from typing import Union, List, Optional
import numpy as np

# Assuming ewma is in utils.
try:
    from RiskLabAI.utils.ewma import ewma 
except ImportError:
    # Fallback if ewma is not in utils (as seen in older files)
    def ewma(array: np.ndarray, window: int) -> np.ndarray:
        """Placeholder EWMA function."""
        if array.size == 0:
            return np.array([np.nan])
        return pd.Series(array).ewm(span=window).mean().values

from RiskLabAI.data.structures.abstract_bars import AbstractBars
from RiskLabAI.utils.constants import *

class AbstractInformationDrivenBars(AbstractBars):
    """
    Abstract class for Information-Driven Bars (Imbalance and Run).

    Implements the shared logic for bars that sample based on
    dynamic thresholds derived from tick imbalances.

    Reference:
        Pages 29-32, Advances in Financial Machine Learning.
    """

    def __init__(
        self,
        bar_type: str,
        window_size_for_expected_n_ticks_estimation: Optional[int],
        initial_estimate_of_expected_n_ticks_in_bar: int,
        window_size_for_expected_imbalance_estimation: int,
    ):
        """
        Constructor.

        Parameters
        ----------
        bar_type : str
            e.g., 'dollar_imbalance', 'tick_run', etc.
        window_size_for_expected_n_ticks_estimation : int, optional
            Window size for EWMA of E[T]. (None for Fixed bars).
        initial_estimate_of_expected_n_ticks_in_bar : int
            Initial guess for E[T].
        window_size_for_expected_imbalance_estimation : int
            Window size for EWMA of E[b].
        """
        super().__init__(bar_type)
        self.information_driven_bars_statistics = {
            EXPECTED_TICKS_NUMBER: float(
                initial_estimate_of_expected_n_ticks_in_bar
            ),
            EXPECTED_IMBALANCE_WINDOW: window_size_for_expected_imbalance_estimation,
        }
        self.window_size_for_expected_n_ticks_estimation = (
            window_size_for_expected_n_ticks_estimation
        )

    def _ewma_expected_imbalance(
        self, array: list, window: int, warm_up: bool = False
    ) -> float:
        """
        Calculate the EWMA of the expected imbalance.

        Parameters
        ----------
        array : list
            List of observed imbalances.
        window : int
            EWMA window.
        warm_up : bool, default=False
            If True, wait for `E[T]` ticks before calculating.

        Returns
        -------
        float
            The EWMA of the expected imbalance.
        """
        if warm_up:
            expected_ticks = self.information_driven_bars_statistics[
                EXPECTED_TICKS_NUMBER
            ]
            if np.isnan(expected_ticks) or len(array) < expected_ticks:
                return np.nan

        ewma_window = int(min(len(array), window))
        if ewma_window == 0:
            return np.nan

        return ewma(
            np.array(array[-ewma_window:], dtype=float), window=ewma_window
        )[-1]

    def _imbalance_at_tick(
        self, price: float, signed_tick: int, volume: float
    ) -> float:
        """
        Calculate the imbalance (theta) for the current tick.

        Parameters
        ----------
        price : float
            Current tick price.
        signed_tick : int
            Current tick rule (1, -1, 0).
        volume : float
            Current tick volume.

        Returns
        -------
        float
            The calculated imbalance.
        """
        if self.bar_type in ("tick_imbalance", "tick_run"):
            imbalance = float(signed_tick)
        elif self.bar_type in ("dollar_imbalance", "dollar_run"):
            imbalance = float(signed_tick * volume * price)
        elif self.bar_type in ("volume_imbalance", "volume_run"):
            imbalance = float(signed_tick * volume)
        else:
            raise ValueError(f"Unknown bar_type for imbalance: {self.bar_type}")

        return imbalance

    @abstractmethod
    def _expected_number_of_ticks(self) -> float:
        """
        Abstract method to update the expected number of ticks (E[T]).
        This is implemented differently for "fixed" vs. "expected" bars.
        """
        pass