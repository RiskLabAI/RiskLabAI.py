from typing import Tuple, Any
import numpy as np

from RiskLabAI.data.structures.base import AbstractBar


class TimeBar(AbstractBar):
    """
    Concrete class of TimeBars logic
    """

    def __init__(self, resolution_type: str, resolution_units: int):
        """
        TimeBars constructor function

        :param resolution_type: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S'].
        :param resolution_units: (int) Number of days, minutes, etc.
        """

        AbstractBar.__init__(self, bar_type="time")

        self.resolution_to_n_seconds = {
            'S': 1,
            'MIN': 60,
            'H': 60 * 60,
            'D': 60 * 60 * 24,
            'W': 60 * 60 * 24 * 7,
        }

        self.resolution_type = resolution_type
        self.resolution_units = resolution_units
        self.threshold_in_seconds = self.resolution_units * self.resolution_to_n_seconds[self.resolution_type]
        self.timestamp = None
        self.timestamp_threshold = np.nan

    def _bar_construction_condition(self, threshold) -> bool:
        return self.timestamp < threshold

    def _compute_tick_data(self, date_time, price, volume, tick_rule) -> Tuple[float, Any]:
        self.timestamp_threshold = ((int(float(date_time.timestamp())) // self.threshold_in_seconds + 1)
                                    * self.threshold_in_seconds)
        # initialize self.timestamp first time
        threshold = self.timestamp_threshold
        if self.timestamp is None:
            self.timestamp = self.timestamp_threshold
        return threshold, None

    def _after_construction_process(self, date_time, price, volume, tick_rule, other_data: Any) -> None:
        self.timestamp = self.timestamp_threshold
