"""
Abstract base class for Imbalance Bars (Fixed and Expected).
"""

from abc import abstractmethod
from typing import Union, List, Any, Iterable, Optional
import numpy as np

from RiskLabAI.data.structures.abstract_information_driven_bars import (
    AbstractInformationDrivenBars
)
from RiskLabAI.data.structures.abstract_bars import TickData
from RiskLabAI.utils.constants import *


class AbstractImbalanceBars(AbstractInformationDrivenBars):
    """
    Abstract class for Imbalance Bars (Fixed and Expected).

    Implements the bar sampling logic based on cumulative imbalance (theta)
    exceeding a dynamic threshold E[T] * |E[b_t]|.

    Reference:
        Pages 29-30, Advances in Financial Machine Learning.
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

        self.imbalance_bars_statistics = {
            CUMULATIVE_θ: 0.0,
            EXPECTED_IMBALANCE: np.nan,
            PREVIOUS_BARS_N_TICKS_LIST: [],
            PREVIOUS_TICK_IMBALANCES_LIST: [],
        }

        self.analyse_thresholds = [] if analyse_thresholds else None

    def construct_bars_from_data(self, data: Iterable[TickData]) -> List[List[Any]]:
        """
        Constructs imbalance bars from input tick data.
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
            self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST].append(
                imbalance
            )
            self.imbalance_bars_statistics[CUMULATIVE_θ] += imbalance

            # Warm-up E[b] if it's the first time
            if np.isnan(self.imbalance_bars_statistics[EXPECTED_IMBALANCE]):
                self.imbalance_bars_statistics[
                    EXPECTED_IMBALANCE
                ] = self._ewma_expected_imbalance(
                    self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST],
                    self.information_driven_bars_statistics[
                        EXPECTED_IMBALANCE_WINDOW
                    ],
                    warm_up=True,
                )

            if self.analyse_thresholds is not None:
                stats = {
                    **self.base_statistics,
                    **self.information_driven_bars_statistics,
                    **self.imbalance_bars_statistics,
                    'timestamp': date_time
                }
                self.analyse_thresholds.append(stats)

            # Calculate threshold and check condition
            expected_ticks = self.information_driven_bars_statistics[
                EXPECTED_TICKS_NUMBER
            ]
            expected_imbalance = self.imbalance_bars_statistics[
                EXPECTED_IMBALANCE
            ]
            
            if np.isnan(expected_ticks) or np.isnan(expected_imbalance):
                threshold = np.inf
            else:
                threshold = expected_ticks * np.abs(expected_imbalance)

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

                # Store T for E[T] update
                self.imbalance_bars_statistics[
                    PREVIOUS_BARS_N_TICKS_LIST
                ].append(self.base_statistics[CUMULATIVE_TICKS])

                # Update E[T]
                self.information_driven_bars_statistics[
                    EXPECTED_TICKS_NUMBER
                ] = self._expected_number_of_ticks()

                # Update E[b]
                self.imbalance_bars_statistics[
                    EXPECTED_IMBALANCE
                ] = self._ewma_expected_imbalance(
                    self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST],
                    self.information_driven_bars_statistics[
                        EXPECTED_IMBALANCE_WINDOW
                    ],
                    warm_up=False,
                )

                # Reset cached fields
                self._reset_cached_fields()

        return bars_list

    def _bar_construction_condition(self, threshold: float) -> bool:
        """Check if cumulative imbalance |theta| exceeds the threshold."""
        if np.isnan(threshold) or np.isinf(threshold):
            return False
            
        cumulative_theta = self.imbalance_bars_statistics[CUMULATIVE_θ]
        return np.abs(cumulative_theta) >= threshold

    def _reset_cached_fields(self):
        """Reset base fields and cumulative theta."""
        super()._reset_cached_fields()
        self.imbalance_bars_statistics[CUMULATIVE_θ] = 0.0

    @abstractmethod
    def _expected_number_of_ticks(self) -> float:
        """Calculate E[T] when a new bar is sampled."""
        pass