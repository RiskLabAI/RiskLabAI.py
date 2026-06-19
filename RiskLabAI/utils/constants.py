DATE_TIME = "Date Time"
TIMESTAMP = "Timestamp"
TICK_NUMBER = "Tick Number"
OPEN_PRICE = "Open"
HIGH_PRICE = "High"
LOW_PRICE = "Low"
CLOSE_PRICE = "Close"

CUMULATIVE_TICKS = "Cumulative Ticks"
CUMULATIVE_DOLLAR = "Cumulative Dollar Value"
THRESHOLD = "Threshold"

CUMULATIVE_VOLUME = "Cumulative Volume"
CUMULATIVE_BUY_VOLUME = "Cumulative Buy Volume"
CUMULATIVE_SELL_VOLUME = "Cumulative Sell Volume"

CUMULATIVE_THETA = "Cumulative θ"
CUMULATIVE_BUY_THETA = "Cumulative Buy θ"
CUMULATIVE_SELL_THETA = "Cumulative Sell θ"

# Deprecated non-ASCII aliases (removed in 2.1.0). The string *values* are
# unchanged, so stored data and internal dict keys are unaffected; only the
# Python identifier you import differs. These are intentionally excluded from
# ``__all__`` so ``RiskLabAI.utils`` can warn on access via its ``__getattr__``.
CUMULATIVE_θ = CUMULATIVE_THETA
CUMULATIVE_BUY_θ = CUMULATIVE_BUY_THETA
CUMULATIVE_SELL_θ = CUMULATIVE_SELL_THETA

EXPECTED_IMBALANCE = "expected_imbalance"
EXPECTED_TICKS_NUMBER = "exp_num_ticks"

EXPECTED_BUY_IMBALANCE = "exp_imbalance_buy"
EXPECTED_SELL_IMBALANCE = "exp_imbalance_sell"
EXPECTED_BUY_TICKS_PROPORTION = "exp_buy_ticks_proportion"
BUY_TICKS_NUMBER = "buy_ticks_num"

N_TICKS_ON_BAR_FORMATION = "Number of ticks while bar is formed."

PREVIOUS_TICK_RULE = "Previous tick rule"
EXPECTED_IMBALANCE_WINDOW = "Expected Imbalance Window"

PREVIOUS_BARS_N_TICKS_LIST = "List of previous bars number of ticks"
PREVIOUS_TICK_IMBALANCES_LIST = "List of previous tick imbalances"

PREVIOUS_TICK_IMBALANCES_BUY_LIST = "List of previous (buy) tick imbalances"
PREVIOUS_TICK_IMBALANCES_SELL_LIST = "List of previous (sell) tick imbalances"

PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST = "List of previous bars buy ticks proportion"

N_PREVIOUS_BARS_FOR_EXPECTED_N_TICKS_ESTIMATION = "Window size for E[T]"

# Canonical public names. The deprecated non-ASCII θ aliases above are
# deliberately omitted so that `from .constants import *` does not bind them in
# RiskLabAI.utils (whose __getattr__ then warns on access). Removed in 2.1.0.
__all__ = [
    "DATE_TIME",
    "TIMESTAMP",
    "TICK_NUMBER",
    "OPEN_PRICE",
    "HIGH_PRICE",
    "LOW_PRICE",
    "CLOSE_PRICE",
    "CUMULATIVE_TICKS",
    "CUMULATIVE_DOLLAR",
    "THRESHOLD",
    "CUMULATIVE_VOLUME",
    "CUMULATIVE_BUY_VOLUME",
    "CUMULATIVE_SELL_VOLUME",
    "CUMULATIVE_THETA",
    "CUMULATIVE_BUY_THETA",
    "CUMULATIVE_SELL_THETA",
    "EXPECTED_IMBALANCE",
    "EXPECTED_TICKS_NUMBER",
    "EXPECTED_BUY_IMBALANCE",
    "EXPECTED_SELL_IMBALANCE",
    "EXPECTED_BUY_TICKS_PROPORTION",
    "BUY_TICKS_NUMBER",
    "N_TICKS_ON_BAR_FORMATION",
    "PREVIOUS_TICK_RULE",
    "EXPECTED_IMBALANCE_WINDOW",
    "PREVIOUS_BARS_N_TICKS_LIST",
    "PREVIOUS_TICK_IMBALANCES_LIST",
    "PREVIOUS_TICK_IMBALANCES_BUY_LIST",
    "PREVIOUS_TICK_IMBALANCES_SELL_LIST",
    "PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST",
    "N_PREVIOUS_BARS_FOR_EXPECTED_N_TICKS_ESTIMATION",
]
