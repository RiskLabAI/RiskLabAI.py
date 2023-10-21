"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from RiskLabAI.utils.constants import *


class AbstractBars(ABC):
    """
    Abstract class that contains the base properties which are shared between the subtypes.
    This class subtypes are as follows:
        1- AbstractImbalanceBars
        2- AbstractRunBars
        3- StandardBars
        4- TimeBars
    """

    def __init__(self, bar_type: str):
        """
        AbstractBars constructor function
        :param bar_type: type of bar. e.g. time_bars, expected_dollar_imbalance_bars, fixed_tick_run_bars, volume_standard_bars etc.
        """

        # main fields
        self.bar_type = bar_type
        self.tick_counter = 0

        # to cached fields
        self.previous_tick_price = None
        self.open_price, self.close_price = None, None
        self.high_price, self.low_price = -np.inf, np.inf
        self.base_statistics = {
            PREVIOUS_TICK_RULE: 0,
            CUMULATIVE_TICKS: 0,
            CUMULATIVE_DOLLAR: 0,
            CUMULATIVE_VOLUME: 0,
            CUMULATIVE_BUY_VOLUME: 0,
            N_TICKS_ON_BAR_FORMATION: 0,
        }

    #todo: don't we need a pass here?
    @abstractmethod
    def construct_bars_from_data(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        This function are implemented by all concrete or abstract subtypes. The function is used to construct bars from
        input ticks data.
        :param data: tabular data that contains date_time, price, and volume columns
        :return: constructed bars
        """

        pass

    def update_base_fields(self, price: float, tick_rule: int, volume: float):
        """
        Update the base fields (that all bars have them.) with price, tick rule and volume of current tick
        :param price: price of current tick
        :param tick_rule: tick rule of current tick computed before
        :param volume: volume of current tick
        :return:
        """

        dollar_value = price * volume

        self.open_price = price if self.open_price is None else self.open_price

        # update high low prices
        self.high_price, self.low_price = self._high_and_low_price_update(price)

        # bar base statistics update
        self.base_statistics[CUMULATIVE_TICKS] += 1
        self.base_statistics[CUMULATIVE_DOLLAR] += dollar_value
        self.base_statistics[CUMULATIVE_VOLUME] += volume

        if tick_rule == 1:
            self.base_statistics[CUMULATIVE_BUY_VOLUME] += volume

    @abstractmethod
    def _bar_construction_condition(self, threshold) -> bool:
        """
        Compute the condition of whether next bar should sample with current and previous tick datas or not.
        :return: whether next bar should form with current and previous tick datas or not.
        """
        pass

    def _reset_cached_fields(self):
        """
        This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
        :return:
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf

        self.base_statistics[CUMULATIVE_TICKS] = 0
        self.base_statistics[CUMULATIVE_DOLLAR] = 0
        self.base_statistics[CUMULATIVE_VOLUME] = 0
        self.base_statistics[CUMULATIVE_BUY_VOLUME] = 0

    def _tick_rule(self, price: float = 0) -> int:
        """
        Compute the tick rule term as explained on page 29 of Advances in Financial Machine Learning
        :param price: price of current tick
        :return: tick rule
        """

        tick_difference = price - self.previous_tick_price if self.previous_tick_price is not None else 0

        if tick_difference != 0:
            self.base_statistics[PREVIOUS_TICK_RULE] = signed_tick = np.sign(tick_difference)
        else:
            signed_tick = self.base_statistics[PREVIOUS_TICK_RULE]

        self.previous_tick_price = price  

        return signed_tick

    def _high_and_low_price_update(self, price: float) -> Tuple[float, float]:
        """
        Update the high and low prices using the current tick price.
        :param price: price of current tick
        :return: updated high and low prices
        """

        high_price = max(price, self.high_price)
        low_price = min(price, self.low_price)

        return high_price, low_price

    def _construct_next_bar(
            self,
            date_time: str,
            tick_index: int,
            price: float,
            high_price: float,
            low_price: float,
            threshold: float,
    ) -> list:
        """
        sample next bar, given ticks data. the bar's fields are as follows:
            1- date_time
            2- open
            3- high
            4- low
            5- close
            6- cumulative_volume: total cumulative volume of to be constructed bar ticks
            7- cumulative_buy_volume: total cumulative buy volume of to be constructed bar ticks
            8- cumulative_ticks total cumulative ticks number of to be constructed bar ticks
            9- cumulative_dollar_value total cumulative dollar value (price * volume) of to be constructed bar ticks

        the bar will have appended to the total list of sampled bars.

        :param date_time: timestamp of the to be constructed bar
        :param tick_index:
        :param price: price of last tick of to be constructed bar (used as close price)
        :param high_price: highest price of ticks in the period of bar sampling process
        :param low_price: lowest price of ticks in the period of bar sampling process
        :return: sampled bar
        """

        open_price, close_price = self.open_price, price
        low_price, high_price = min(low_price, open_price), max(high_price, open_price)

        cumulative_ticks = self.base_statistics[CUMULATIVE_TICKS]
        cumulative_dollar_value = self.base_statistics[CUMULATIVE_DOLLAR]

        cumulative_volume = self.base_statistics[CUMULATIVE_VOLUME]
        cumulative_buy_volume = self.base_statistics[CUMULATIVE_BUY_VOLUME]
        cumulative_sell_volume = cumulative_volume - cumulative_buy_volume
        
        next_bar = [
            date_time,
            tick_index,
            open_price, high_price, low_price, close_price,
            cumulative_volume, cumulative_buy_volume, cumulative_sell_volume,
            cumulative_ticks,
            cumulative_dollar_value,
            threshold
        ]

        return next_bar
