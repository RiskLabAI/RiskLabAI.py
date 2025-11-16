"""
Implements Standard Bars (Tick, Volume, Dollar).
"""

from typing import Union, List, Any, Iterable
import numpy as np
from RiskLabAI.data.structures.abstract_bars import AbstractBars, TickData

class StandardBars(AbstractBars):
    """
    Concrete class for Standard Bars (Tick, Volume, Dollar).

    Generates a new bar whenever a cumulative threshold of ticks,
    volume, or dollars is reached.
    """

    def __init__(self, bar_type: str, threshold: float = 50000):
        """
        StandardBars constructor.

        Parameters
        ----------
        bar_type : str
            Type of bar. Must be one of:
            `CUMULATIVE_TICKS`, `CUMULATIVE_VOLUME`, `CUMULATIVE_DOLLAR`.
        threshold : float, default=50000
            The threshold value to trigger bar construction.
        """
        super().__init__(bar_type)
        self.threshold = threshold

    def construct_bars_from_data(self, data: Iterable[TickData]) -> List[List[Any]]:
        """
        Constructs standard bars from input tick data.

        Parameters
        ----------
        data : Iterable[TickData]
            An iterable (list, tuple, generator) of tick data.
            Each tick is (date_time, price, volume).

        Returns
        -------
        List[List[Any]]
            A list of the constructed standard bars.
        """
        bars_list = []
        
        # Keep track of last timestamp for final bar
        date_time = None 
        
        for tick_data in data:
            self.tick_counter += 1

            # Unpack data
            date_time, price, volume = tick_data[0], tick_data[1], tick_data[2]
            
            # Update common fields
            tick_rule = self._tick_rule(price)
            self.update_base_fields(price, tick_rule, volume)
            self.close_price = price # Update close price continuously

            # Check if bar construction condition is met
            if self._bar_construction_condition(self.threshold):
                next_bar = self._construct_next_bar(
                    date_time,
                    self.tick_counter,
                    self.close_price,
                    self.high_price,
                    self.low_price,
                    self.threshold,
                )
                bars_list.append(next_bar)
                
                # Reset cached fields for the next bar
                self._reset_cached_fields()

        return bars_list

    def _bar_construction_condition(self, threshold: float) -> bool:
        """
        Check if the cumulative value of the `bar_type` has
        exceeded the threshold.
        """
        return self.base_statistics[self.bar_type] >= threshold