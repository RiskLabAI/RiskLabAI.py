"""
Tests for controller/bars_initializer.py
"""

import pytest
from .test_bars_initializer import BarsInitializerController

# Import all bar types to check instance
from RiskLabAI.data.structures.imbalance_bars import (
    ExpectedImbalanceBars, FixedImbalanceBars
)
from RiskLabAI.data.structures.run_bars import ExpectedRunBars, FixedRunBars
from RiskLabAI.data.structures.standard_bars import StandardBars
from RiskLabAI.data.structures.time_bars import TimeBars

@pytest.fixture
def controller():
    """Fixture for the BarsInitializerController."""
    return BarsInitializerController()

# Get all method names from the controller's map
bar_types_to_test = list(BarsInitializerController().method_name_to_method.keys())
expected_classes = {
    "expected_dollar_imbalance_bars": ExpectedImbalanceBars,
    "expected_volume_imbalance_bars": ExpectedImbalanceBars,
    "expected_tick_imbalance_bars": ExpectedImbalanceBars,
    "fixed_dollar_imbalance_bars": FixedImbalanceBars,
    "fixed_volume_imbalance_bars": FixedImbalanceBars,
    "fixed_tick_imbalance_bars": FixedImbalanceBars,
    "expected_dollar_run_bars": ExpectedRunBars,
    "expected_volume_run_bars": ExpectedRunBars,
    "expected_tick_run_bars": ExpectedRunBars,
    "fixed_dollar_run_bars": FixedRunBars,
    "fixed_volume_run_bars": FixedRunBars,
    "fixed_tick_run_bars": FixedRunBars,
    "dollar_standard_bars": StandardBars,
    "volume_standard_bars": StandardBars,
    "tick_standard_bars": StandardBars,
    "time_bars": TimeBars,
}

@pytest.mark.parametrize("bar_method_name", bar_types_to_test)
def test_all_bar_initializers(controller, bar_method_name):
    """
    Test that every method in the controller map successfully
    creates a bar object of the expected type.
    """
    # Get the actual initialization method from the controller
    init_method = controller.method_name_to_method[bar_method_name]
    
    # Call the method (e.g., controller.initialize_time_bars())
    bar_instance = init_method()
    
    # Check that it's an instance of the correct base class
    expected_class = expected_classes[bar_method_name]
    assert isinstance(bar_instance, expected_class)
    
    # Check specific bar types for imbalance/run bars
    if bar_method_name.endswith("_imbalance_bars"):
        assert bar_method_name.split("_")[1] in bar_instance.bar_type
    elif bar_method_name.endswith("_run_bars"):
        assert bar_method_name.split("_")[1] in bar_instance.bar_type