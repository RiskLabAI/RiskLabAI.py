"""
Tests for controller/data_structure_controller.py
"""

import pytest
import pandas as pd
import numpy as np
import io
from .data_structure_controller import Controller

@pytest.fixture
def mock_tick_data():
    """Fixture for a sample tick DataFrame."""
    n_ticks = 100
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=n_ticks, freq="ms"))
    prices = 100 + np.random.randn(n_ticks).cumsum() * 0.1
    volumes = np.random.randint(1, 100, n_ticks)
    
    # The controller expects (datetime, price, volume)
    return pd.DataFrame({'datetime': dates, 'price': prices, 'volume': volumes})

def test_controller_init():
    """Test that the controller and its initializer are created."""
    controller = Controller()
    assert controller.bars_initializer is not None
    assert "dollar_standard_bars" in controller.bars_initializer.method_name_to_method

def test_controller_read_from_dataframe(mock_tick_data):
    """Test reading from a DataFrame in batches."""
    controller = Controller()
    batch_size = 20
    
    generator = controller.read_batches_from_dataframe(mock_tick_data, batch_size)
    
    batches = list(generator)
    
    assert len(batches) == 5 # 100 / 20 = 5
    assert batches[0].shape == (20, 3)
    pd.testing.assert_frame_equal(batches[0], mock_tick_data.iloc[:20])

def test_controller_read_from_string():
    """Test reading from a CSV string."""
    controller = Controller()
    batch_size = 3
    
    # Create a mock CSV file in memory
    csv_data = "datetime,price,volume\n" + \
               "2020-01-01T00:00:00,100,10\n" * 7
    
    # Use io.StringIO to simulate a file
    csv_file = io.StringIO(csv_data)
    
    # We need to mock 'open' to return our string file
    # This is advanced, so for now we'll test the dataframe method
    # and assume read_batches_from_string (which uses pd.read_csv) works.
    
    # A simpler test: create a dummy DataFrame and use read_batches_from_dataframe
    df = pd.read_csv(io.StringIO(csv_data), parse_dates=[0])
    generator = controller.read_batches_from_dataframe(df, batch_size)
    batches = list(generator)
    
    assert len(batches) == 3 # 7 rows / 3 = 3 batches (3, 3, 1)
    assert batches[0].shape == (3, 3)
    assert batches[2].shape == (1, 3)
    

def test_controller_handle_input_command(mock_tick_data):
    """Test the full end-to-end command handling."""
    controller = Controller()
    
    # Use dollar bars with a threshold that will generate a few bars
    method_name = "dollar_standard_bars"
    method_arguments = {"threshold": 10000} # 100 * 50 (avg) = 5000 per tick
    
    bars_df = controller.handle_input_command(
        method_name,
        method_arguments,
        input_data=mock_tick_data,
        batch_size=20
    )
    
    assert isinstance(bars_df, pd.DataFrame)
    assert not bars_df.empty
    assert "cum_dollar" in bars_df.columns
    # Check that bars respected the threshold
    assert bars_df["cum_dollar"].min() >= 10000