"""
Tests for features/structural_breaks/structural_breaks.py
"""

import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from RiskLabAI.features.structural_breaks.structural_breaks import (
    lag_dataframe,
    prepare_data,
    compute_beta,
    get_bsadf_statistic
)

@pytest.fixture
def sample_series():
    """A simple series for testing."""
    return pd.DataFrame({'price': [1.0, 1.2, 1.1, 1.3, 1.5, 1.4]})

@pytest.fixture
def random_walk_series():
    """A non-stationary random walk."""
    rng = np.random.default_rng(42)
    log_price = np.log(100 + rng.normal(0, 1, 100).cumsum())
    return pd.DataFrame({'log_price': log_price})

def test_lag_dataframe(sample_series):
    """Test the lag_dataframe function."""
    lags = 2
    df = lag_dataframe(sample_series, lags)
    
    # Should create columns for lags 0, 1, 2
    assert 'price_0' in df.columns
    assert 'price_1' in df.columns
    assert 'price_2' in df.columns
    
    # Check values
    assert np.isclose(df['price_0'].iloc[2], 1.1)
    assert np.isclose(df['price_1'].iloc[2], 1.2)
    assert np.isclose(df['price_2'].iloc[2], 1.0)
    assert pd.isna(df['price_2'].iloc[1])

def test_prepare_data(sample_series):
    """Test the prepare_data function."""
    lags = 1
    # series = [1.0, 1.2, 1.1, 1.3, 1.5, 1.4]
    # diff = [nan, 0.2, -0.1, 0.2, 0.2, -0.1]
    # x = lag_dataframe(diff, 1)
    #   diff_0 = [nan, 0.2, -0.1, 0.2, 0.2, -0.1]
    #   diff_1 = [nan, nan, 0.2, -0.1, 0.2, 0.2]
    # x (dropna) = (index 2-5)
    #   [ -0.1, 0.2 ]
    #   [ 0.2, -0.1 ]
    #   [ 0.2, 0.2 ]
    #   [ -0.1, 0.2 ]
    # x (replace col 0 with level)
    #   [ 1.1, 0.2 ]
    #   [ 1.3, -0.1 ]
    #   [ 1.5, 0.2 ]
    #   [ 1.4, 0.2 ]
    # y = diff (index 2-5)
    #   [ -0.1, 0.2, 0.2, -0.1 ]
    
    y, x = prepare_data(sample_series, constant='c', lags=lags)
    
    assert y.shape == (4, 1)
    assert x.shape == (4, 3) # level, lag 1 diff, constant
    
    expected_y = np.array([[-0.1], [0.2], [0.2], [-0.1]])
    expected_x = np.array([
        [1.2,  0.2, 1.0],  
        [1.1, -0.1, 1.0],  
        [1.3,  0.2, 1.0],  
        [1.5,  0.2, 1.0]   
    ])
    
    assert np.allclose(y, expected_y)
    assert np.allclose(x, expected_x)

def test_compute_beta_bugfix():
    """
    Test compute_beta against statsmodels to verify the bugfix.
    """
    # 1. Prepare data
    y_vec = np.array([1, 2, 3, 4, 5], dtype=float)
    x_vec = np.array([1.1, 1.9, 3.0, 4.1, 4.9], dtype=float)
    x_mat = sm.add_constant(x_vec) # [const, x1]
    y_vec = y_vec.reshape(-1, 1)
    
    # 2. Get correct result from statsmodels
    model = sm.OLS(y_vec, x_mat).fit()
    sm_betas = model.params.reshape(-1, 1)
    sm_vcov = model.cov_params()
    
    # 3. Get result from our function
    my_betas, my_vcov = compute_beta(y_vec, x_mat)
    
    # 4. Compare
    assert np.allclose(my_betas, sm_betas)
    assert np.allclose(my_vcov, sm_vcov)
    


def test_adf_function(random_walk_series):
    """Test the main ADF loop."""
    results = get_bsadf_statistic(
        log_price=random_walk_series,
        min_sample_length=20,
        constant='c',
        lags=1
    )
    
    assert 'Time' in results
    assert 'gsadf' in results
    assert isinstance(results['Time'], int)
    assert isinstance(results['gsadf'], float)
    assert np.isfinite(results['gsadf'])