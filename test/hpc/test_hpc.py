"""
Tests for hpc/hpc.py
"""

import pytest
import pandas as pd
import numpy as np
from .hpc import (
    linear_partitions,
    nested_partitions,
    process_jobs_sequential,
    process_jobs,
    mp_pandas_obj
)

# --- Test Functions for Parallelism ---

def _test_func_sum(x: int, y: int) -> int:
    """Simple function for testing."""
    return x + y

def _test_func_pandas(molecule: pd.Index, series: pd.Series) -> pd.Series:
    """Test function for mp_pandas_obj."""
    return series.loc[molecule] * 2

# --- Tests ---

def test_linear_partitions():
    """Test linear partitioning."""
    # 100 items, 4 threads
    parts = linear_partitions(num_atoms=100, num_threads=4)
    expected = np.array([0, 25, 50, 75, 100])
    np.testing.assert_array_equal(parts, expected)
    
    # 10 atoms, 12 threads (caps at num_atoms)
    parts_capped = linear_partitions(num_atoms=10, num_threads=12)
    assert len(parts_capped) == 11 # 10 partitions + 1
    assert parts_capped[-1] == 10

def test_nested_partitions():
    """Test nested partitioning."""
    parts = nested_partitions(num_atoms=100, num_threads=4)
    # Partitions are not equal
    assert len(parts) == 5
    assert parts[0] == 0
    assert parts[-1] == 100
    assert parts[1] - parts[0] != parts[2] - parts[1]

def test_process_jobs_sequential():
    """Test the sequential job processor."""
    jobs = [
        {'func': _test_func_sum, 'x': 1, 'y': 2}, # 3
        {'func': _test_func_sum, 'x': 5, 'y': 5}, # 10
    ]
    results = process_jobs_sequential(jobs)
    assert results == [3, 10]

def test_process_jobs_parallel():
    """Test the parallel job processor."""
    jobs = [
        {'func': _test_func_sum, 'x': 1, 'y': 2}, # 3
        {'func': _test_func_sum, 'x': 5, 'y': 5}, # 10
    ]
    results = process_jobs(jobs, num_threads=2)
    assert sorted(results) == [3, 10]

def test_mp_pandas_obj():
    """Test the mp_pandas_obj wrapper."""
    series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
    pd_obj = ('molecule', series.index)
    
    result = mp_pandas_obj(
        _test_func_pandas,
        pd_obj,
        num_threads=2,
        series=series
    )
    
    expected = series * 2
    pd.testing.assert_series_equal(result, expected)