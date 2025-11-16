"""
Implements Time Bars.
"""

from typing import Union, List, Any, Iterable
import numpy as np
import pandas as pd
from RiskLabAI.data.structures.abstract_bars import AbstractBars, TickData

class TimeBars(AbstractBars):
    """
    Concrete class for Time Bars.

    Generates a new bar whenever a fixed time duration has passed.
    """

    def __init__(self, resolution_type: str, resolution_units: int):
        """
        TimeBars constructor.

        Parameters
        ----------
        resolution_type : str
            Type of bar resolution: 'S', 'MIN', 'H', 'D', 'W'.
        resolution_units : int
            Number of units for the resolution (e.g., 5 for 5-minutes).
        """
        super().__init__(bar_type="time")

        self.resolution_to_n_seconds = {
            "S": 1,
            "MIN": 60,
            "H": 3600,
            "D": 86400,
            "W": 604800,
        }

        if resolution_type.upper() not in self.resolution_to_n_seconds:
            raise ValueError(f"Invalid resolution_type. Use one of {list(self.resolution_to_n_seconds.keys())}")
            
        self.resolution_type = resolution_type.upper()
        self.resolution_units = resolution_units
        self.threshold_in_seconds = (
            self.resolution_units * self.resolution_to_n_seconds[self.resolution_type]
        )
        
        self.current_bar_timestamp = np.nan
        self.current_bar_end_timestamp = np.nan

    def construct_bars_from_data(self, data: Iterable[TickData]) -> List[List[Any]]:
        """
        Constructs time bars from input tick data.

        Parameters
        ----------
        data : Iterable[TickData]
            An iterable (list, tuple, generator) of tick data.
            Each tick is (date_time, price, volume). 
            `date_time` must be a pandas Timestamp.

        Returns
        -------
        List[List[Any]]
            A list of the constructed time bars.
        """
        bars_list = []
        for tick_data in data:
            self.tick_counter += 1

            # Unpack data
            date_time, price, volume = tick_data[0], tick_data[1], tick_data[2]
            
            # Get tick timestamp in seconds
            try:
                tick_timestamp_sec = date_time.timestamp()
            except AttributeError:
                raise TypeError(
                    "TimeBars require `date_time` to be a pandas Timestamp "
                    "or datetime object with a .timestamp() method."
                )
            
            # Determine the "floor" timestamp for this bar
            bar_start_timestamp_sec = (
                int(tick_timestamp_sec // self.threshold_in_seconds)
                * self.threshold_in_seconds
            )
            
            # Initialize first bar
            if np.isnan(self.current_bar_timestamp):
                self.current_bar_timestamp = bar_start_timestamp_sec
                self.current_bar_end_timestamp = (
                    bar_start_timestamp_sec + self.threshold_in_seconds
                )
            
            # Check if this tick belongs to a new bar
            if self._bar_construction_condition(tick_timestamp_sec):
                # Construct the *previous* bar
                bar_end_time = pd.to_datetime(self.current_bar_end_timestamp, unit='s')
                
                next_bar = self._construct_next_bar(
                    bar_end_time,
                    self.tick_counter - 1, # Index of the *previous* tick
                    self.close_price,      # Close price from *previous* tick
                    self.high_price,
                    self.low_price,
                    self.current_bar_end_timestamp,
                )
                bars_list.append(next_bar)

                # Reset for the new bar
                self._reset_cached_fields()
                self.current_bar_timestamp = bar_start_timestamp_sec
                self.current_bar_end_timestamp = (
                    bar_start_timestamp_sec + self.threshold_in_seconds
                )

            # Update fields with current tick data
            tick_rule = self._tick_rule(price)
            self.update_base_fields(price, tick_rule, volume)
            self.close_price = price
            
        return bars_list

    def _bar_construction_condition(self, tick_timestamp_sec: float) -> bool:
        """
        Check if the current tick's timestamp has crossed
        the end-time of the current bar.
        """
        return tick_timestamp_sec >= self.current_bar_end_timestamp