"""
Abstract base class for Run Bars (Fixed and Expected).
"""

from abc import abstractmethod
from typing import Union, List, Any, Iterable, Optional
import numpy as np
import pandas as pd # Added for ewma fallback

from RiskLabAI.data.structures.abstract_information_driven_bars import (
    AbstractInformationDrivenBars
)
from RiskLabAI.data.structures.abstract_bars import TickData

try:
    from RiskLabAI.utils.ewma import ewma 
except ImportError:
    # Fallback if ewma is not in utils (as seen in older files)
    def ewma(array: np.ndarray, window: int) -> np.ndarray:
        """Placeholder EWMA function."""
        if array.size == 0:
            return np.array([np.nan])
        return pd.Series(array).ewm(span=window).mean().values

from RiskLabAI.utils.constants import *


class AbstractRunBars(AbstractInformationDrivenBars):
    """
    Abstract class for Run Bars (Fixed and Expected).
    (Docstrings same as original)
    """

    def __init__(
        self,
        bar_type: str,
        window_size_for_expected_n_ticks_estimation: Optional[int],
        window_size_for_expected_imbalance_estimation: int,
        initial_estimate_of_expected_n_ticks_in_bar: int,
        analyse_thresholds: bool,
    ):
        """
        Constructor.
        (Parameters same as original)
        """
        super().__init__(
            bar_type,
            window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation,
        )

        self.run_bars_statistics = {
            CUMULATIVE_BUY_θ: 0.0,
            CUMULATIVE_SELL_θ: 0.0,
            EXPECTED_BUY_IMBALANCE: np.nan,
            EXPECTED_SELL_IMBALANCE: np.nan,
            EXPECTED_BUY_TICKS_PROPORTION: np.nan,
            BUY_TICKS_NUMBER: 0,
            PREVIOUS_BARS_N_TICKS_LIST: [],
            PREVIOUS_TICK_IMBALANCES_BUY_LIST: [],
            PREVIOUS_TICK_IMBALANCES_SELL_LIST: [],
            PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST: [],
        }

        self.analyse_thresholds = [] if analyse_thresholds else None

    def construct_bars_from_data(self, data: Iterable[TickData]) -> List[List[Any]]:
        """
        Constructs run bars from input tick data.
        (Parameters same as original)
        """
        bars_list = []
        
        # Keep track of last timestamp and threshold
        date_time = None
        threshold = np.inf
        
        for tick_data in data:
            self.tick_counter += 1

            date_time, price, volume = tick_data[0], tick_data[1], tick_data[2]
            
            # Update common fields
            tick_rule = self._tick_rule(price)
            self.update_base_fields(price, tick_rule, volume)
            self.close_price = price

            # Calculate imbalance
            imbalance = self._imbalance_at_tick(price, tick_rule, volume)

            if imbalance > 0:
                self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_BUY_LIST].append(
                    imbalance
                )
                self.run_bars_statistics[CUMULATIVE_BUY_θ] += imbalance
                self.run_bars_statistics[BUY_TICKS_NUMBER] += 1
            elif imbalance < 0:
                self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_SELL_LIST].append(
                    -imbalance
                )
                self.run_bars_statistics[CUMULATIVE_SELL_θ] += -imbalance

            # Warm-up E[theta_buy], E[theta_sell], and P[buy]
            warm_up_stats = [
                self.run_bars_statistics[EXPECTED_BUY_IMBALANCE],
                self.run_bars_statistics[EXPECTED_SELL_IMBALANCE],
                self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION]
            ]
            
            if np.isnan(warm_up_stats).any():
                self.run_bars_statistics[
                    EXPECTED_BUY_IMBALANCE
                ] = self._ewma_expected_imbalance(
                    self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_BUY_LIST],
                    self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
                    warm_up=True
                )
                self.run_bars_statistics[
                    EXPECTED_SELL_IMBALANCE
                ] = self._ewma_expected_imbalance(
                    self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_SELL_LIST],
                    self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
                    warm_up=True
                )
                
                # Update P[buy]
                if self.base_statistics[CUMULATIVE_TICKS] > 0:
                    buy_ticks_num = self.run_bars_statistics[BUY_TICKS_NUMBER]
                    cum_ticks = self.base_statistics[CUMULATIVE_TICKS]
                    self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION] = (
                        buy_ticks_num / cum_ticks
                    )

            if self.analyse_thresholds is not None:
                stats = {
                    **self.base_statistics,
                    **self.information_driven_bars_statistics,
                    **self.run_bars_statistics,
                    'timestamp': date_time
                }
                self.analyse_thresholds.append(stats)
            
            # Calculate threshold and check condition
            threshold = self._calculate_run_threshold()

            if self._bar_construction_condition(threshold):
                next_bar = self._construct_next_bar(
                    date_time,
                    self.tick_counter,
                    self.close_price,
                    self.high_price,
                    self.low_price,
                    threshold,
                )
                bars_list.append(next_bar)

                # Store T and P[buy] for E[T] and E[P[buy]] updates
                cum_ticks = self.base_statistics[CUMULATIVE_TICKS]
                buy_ticks_num = self.run_bars_statistics[BUY_TICKS_NUMBER]
                
                self.run_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST].append(
                    cum_ticks
                )
                
                # Avoid division by zero if bar has 0 ticks (should be rare)
                buy_proportion = (buy_ticks_num / cum_ticks) if cum_ticks > 0 else 0
                self.run_bars_statistics[
                    PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST
                ].append(buy_proportion)

                # Update E[T]
                self.information_driven_bars_statistics[
                    EXPECTED_TICKS_NUMBER
                ] = self._expected_number_of_ticks()

                # Update E[P[buy]]
                window = self.window_size_for_expected_n_ticks_estimation or \
                        self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW]

                prob_buy_list = self.run_bars_statistics[
                    PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST
                ]
                
                self.run_bars_statistics[
                    EXPECTED_BUY_TICKS_PROPORTION
                ] = ewma(
                    np.array(prob_buy_list[-window:], dtype=float),
                    window,
                )[-1]

                # Update E[theta_buy] and E[theta_sell]
                self.run_bars_statistics[
                    EXPECTED_BUY_IMBALANCE
                ] = self._ewma_expected_imbalance(
                    self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_BUY_LIST],
                    self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW]
                )
                self.run_bars_statistics[
                    EXPECTED_SELL_IMBALANCE
                ] = self._ewma_expected_imbalance(
                    self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_SELL_LIST],
                    self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW]
                )

                # Reset cached fields
                self._reset_cached_fields()

        return bars_list

    def _calculate_run_threshold(self) -> float:
        """Helper function to calculate the dynamic run threshold."""
        e_t = self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]
        e_p_buy = self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION]
        e_theta_buy = self.run_bars_statistics[EXPECTED_BUY_IMBALANCE]
        e_theta_sell = self.run_bars_statistics[EXPECTED_SELL_IMBALANCE]

        if np.isnan([e_t, e_p_buy, e_theta_buy, e_theta_sell]).any():
            return np.inf

        # Threshold = E[T] * max(P[buy] * E[theta_buy], (1-P[buy]) * E[theta_sell])
        buy_threshold = e_p_buy * e_theta_buy
        sell_threshold = (1 - e_p_buy) * e_theta_sell
        
        return e_t * max(buy_threshold, sell_threshold)


    def _bar_construction_condition(self, threshold: float) -> bool:
        """Check if cumulative buy or sell run exceeds the threshold."""
        if np.isinf(threshold) or np.isnan(threshold):
            return False
            
        max_theta = max(
            self.run_bars_statistics[CUMULATIVE_BUY_θ],
            self.run_bars_statistics[CUMULATIVE_SELL_θ],
        )
        return max_theta >= threshold

    def _reset_cached_fields(self):
        """Reset base fields and cumulative run counters."""
        super()._reset_cached_fields()
        self.run_bars_statistics[CUMULATIVE_BUY_θ] = 0.0
        self.run_bars_statistics[CUMULATIVE_SELL_θ] = 0.0
        self.run_bars_statistics[BUY_TICKS_NUMBER] = 0

    @abstractmethod
    def _expected_number_of_ticks(self) -> float:
        """Calculate E[T] when a new bar is sampled."""
        pass