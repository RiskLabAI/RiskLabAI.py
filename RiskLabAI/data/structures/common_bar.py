from typing import Tuple, Any, Union
import numpy as np

from RiskLabAI.data.structures.base import AbstractBar
from RiskLabAI.utils.constants import *


class CommonBar(AbstractBar):
    """
    Concrete class of BasicBars logic. It represents Tick, Volume and Dollar bars.
    The type of bar is defined by the bar_type parameter.
    """

    def __init__(self, bar_type: str, unit_per_bar: int = 0, number_bars: int = 100):
        """
        BasicBars constructor function

        :param bar_type: (str) Type of bar: ['tick', 'volume', 'dollar'].
        """
        assert bar_type in ['tick', 'volume', 'dollar'], "Bar type must be one of ['tick', 'volume', 'dollar']"
        AbstractBar.__init__(self, bar_type=bar_type)

        if bar_type == 'tick':
            self.base_statistic_key = CUMULATIVE_TICKS
        elif bar_type == 'volume':
            self.base_statistic_key = CUMULATIVE_VOLUME
        else:
            self.base_statistic_key = CUMULATIVE_DOLLAR

        self.unit_per_bar = unit_per_bar
        self.number_bars = number_bars

    def _pre_process_data(self, data: Union[list, tuple, np.ndarray]) -> None:
        if self.unit_per_bar == 0:
            if self.bar_type == 'tick' or self.bar_type == 'volume':
                self.unit_per_bar = int(len(data) / self.number_bars)
            elif self.bar_type == 'dollar':
                if isinstance(data, np.ndarray):
                    dollars_total = np.sum(data[:, 1] * data[:, 2])
                    self.unit_per_bar = dollars_total / self.number_bars
                    self.unit_per_bar = round(self.unit_per_bar, -2)
                else:
                    dollar_total = 0
                    for tick in data:
                        date_time, price, volume = tuple(tick)
                        dollar = price * volume
                        dollar_total += dollar
                    self.unit_per_bar = dollar_total / self.number_bars
                    self.unit_per_bar = round(self.unit_per_bar, -2)
            else:
                raise ValueError("Bar type must be one of ['tick', 'volume', 'dollar']")

    def _bar_construction_condition(self, threshold) -> bool:
        return self.base_statistics[self.base_statistic_key] < threshold

    def _compute_tick_data(self, date_time, price, volume, tick_rule) -> Tuple[float, Any]:
        return self.unit_per_bar, None

    def _after_construction_process(self, date_time, price, volume, tick_rule, other_data: Any) -> None:
        pass
