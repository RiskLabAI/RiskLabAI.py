"""
A base class for the various bar types. Includes the logic shared between
classes, to minimise duplicated code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Any, Dict, Iterable, Optional
import numpy as np
from RiskLabAI.utils.constants import *

# Type hint for a single tick: (datetime, price, volume)
TickData = Union[List[Any], Tuple[Any, ...], np.ndarray]

class AbstractBars(ABC):
    """
    Abstract base class for all bar types.

    Contains the base properties and methods shared between all
    bar structures, such as Standard, Time, and Information-Driven bars.
    """

    def __init__(self, bar_type: str):
        """
        AbstractBars constructor.

        Parameters
        ----------
        bar_type : str
            The type of bar, used as a key for statistics.
            e.g., 'dollar', 'volume', 'tick_imbalance', etc.
        """
        # Main fields
        self.bar_type = bar_type
        self.tick_counter = 0  # Global tick counter

        # Cached fields for the current bar being built
        self.previous_tick_price: Optional[float] = None
        self.open_price: Optional[float] = None
        self.close_price: Optional[float] = None
        self.high_price: float = -np.inf
        self.low_price: float = np.inf
        
        self.base_statistics: Dict[str, Union[int, float]] = {
            PREVIOUS_TICK_RULE: 0,
            CUMULATIVE_TICKS: 0,
            CUMULATIVE_DOLLAR: 0,
            CUMULATIVE_VOLUME: 0,
            CUMULATIVE_BUY_VOLUME: 0,
            N_TICKS_ON_BAR_FORMATION: 0,
        }

    @abstractmethod
    def construct_bars_from_data(self, data: Iterable[TickData]) -> List[List[Any]]:
        """
        Constructs bars from input tick data.

        This method must be implemented by all concrete subtypes.

        Parameters
        ----------
        data : Iterable[TickData]
            An iterable (list, tuple, generator) of tick data.
            Each tick is expected to be a tuple/list of
            (date_time, price, volume).

        Returns
        -------
        List[List[Any]]
            A list of the constructed bars, where each bar is a list
            of aggregated values.
        """
        pass

    def update_base_fields(self, price: float, tick_rule: int, volume: float):
        """
        Update the base statistics for the current bar in progress.

        Parameters
        ----------
        price : float
            Price of the current tick.
        tick_rule : int
            Tick rule (1, -1, or 0) of the current tick.
        volume : float
            Volume of the current tick.
        """
        dollar_value = price * volume

        if self.open_price is None:
            self.open_price = price

        # Update high/low prices
        self.high_price, self.low_price = self._high_and_low_price_update(price)

        # Update bar base statistics
        self.base_statistics[CUMULATIVE_TICKS] += 1
        self.base_statistics[CUMULATIVE_DOLLAR] += dollar_value
        self.base_statistics[CUMULATIVE_VOLUME] += volume

        if tick_rule == 1:
            self.base_statistics[CUMULATIVE_BUY_VOLUME] += volume

    @abstractmethod
    def _bar_construction_condition(self, threshold: float) -> bool:
        """
        Check if the condition to sample a new bar has been met.

        Parameters
        ----------
        threshold : float
            The threshold value to check against.

        Returns
        -------
        bool
            True if the bar should be sampled, False otherwise.
        """
        pass

    def _reset_cached_fields(self):
        """
        Reset the cached fields when a new bar is sampled.
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf

        self.base_statistics[CUMULATIVE_TICKS] = 0
        self.base_statistics[CUMULATIVE_DOLLAR] = 0
        self.base_statistics[CUMULATIVE_VOLUME] = 0
        self.base_statistics[CUMULATIVE_BUY_VOLUME] = 0

    def _tick_rule(self, price: float) -> int:
        """
        Compute the tick rule (sign of the price change).

        Reference:
            Page 29, Advances in Financial Machine Learning.

        Parameters
        ----------
        price : float
            Price of the current tick.

        Returns
        -------
        int
            The signed tick (1, -1, or 0).
        """
        if self.previous_tick_price is not None:
            tick_difference = price - self.previous_tick_price
        else:
            tick_difference = 0.0

        if tick_difference != 0:
            signed_tick = int(np.sign(tick_difference))
            self.base_statistics[PREVIOUS_TICK_RULE] = signed_tick
        else:
            # Use the previous non-zero tick rule
            signed_tick = int(self.base_statistics[PREVIOUS_TICK_RULE])

        self.previous_tick_price = price
        return signed_tick

    def _high_and_low_price_update(self, price: float) -> Tuple[float, float]:
        """
        Update the high and low prices for the current bar.

        Parameters
        ----------
        price : float
            Price of the current tick.

        Returns
        -------
        Tuple[float, float]
            (new_high_price, new_low_price)
        """
        high_price = max(price, self.high_price)
        low_price = min(price, self.low_price)
        return high_price, low_price

    def _construct_next_bar(
        self,
        date_time: Any,
        tick_index: int,
        price: float,
        high_price: float,
        low_price: float,
        threshold: float,
    ) -> List[Any]:
        """
        Format and return the newly constructed bar.

        Parameters
        ----------
        date_time : Any
            Timestamp for the bar.
        tick_index : int
            The global tick counter index.
        price : float
            The close price of the bar.
        high_price : float
            The high price of the bar.
        low_price : float
            The low price of the bar.
        threshold : float
            The threshold that triggered this bar's construction.

        Returns
        -------
        List[Any]
            A list containing all aggregated bar data.
        """
        open_price = self.open_price if self.open_price is not None else price
        close_price = price
        low_price = min(low_price, open_price)
        high_price = max(high_price, open_price)

        # Get cumulative statistics
        cumulative_ticks = self.base_statistics[CUMULATIVE_TICKS]
        cumulative_dollar = self.base_statistics[CUMULATIVE_DOLLAR]
        cumulative_volume = self.base_statistics[CUMULATIVE_VOLUME]
        cumulative_buy_volume = self.base_statistics[CUMULATIVE_BUY_VOLUME]
        cumulative_sell_volume = cumulative_volume - cumulative_buy_volume

        next_bar = [
            date_time,
            tick_index,
            open_price,
            high_price,
            low_price,
            close_price,
            cumulative_volume,
            cumulative_buy_volume,
            cumulative_sell_volume,
            cumulative_ticks,
            cumulative_dollar,
            threshold,
        ]

        return next_bar