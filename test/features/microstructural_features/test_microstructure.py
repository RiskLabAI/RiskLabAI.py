"""
Tests for modules in features/microstructural_features/
"""

import numpy as np
import pandas as pd
import pytest
from RiskLabAI.features.microstructural_features.corwin_schultz import (
    beta_estimates,
    gamma_estimates,
    alpha_estimates,
    corwin_schultz_estimator,
)
from RiskLabAI.features.microstructural_features.bekker_parkinson_volatility_estimator import (
    sigma_estimates,
    bekker_parkinson_volatility_estimates,
)

@pytest.fixture
def sample_hl_prices():
    """Fixture for high/low price series."""
    # Using simple data for testing
    high = pd.Series([10.1, 10.3, 10.2, 10.4, 10.5])
    low = pd.Series([9.9, 10.1, 10.0, 10.2, 10.3])
    return high, low

def test_corwin_schultz(sample_hl_prices):
    """Test the Corwin-Schultz estimator end-to-end."""
    high, low = sample_hl_prices
    window = 2
    
    # 1. Beta
    beta = beta_estimates(high, low, window)
    # log(H/L)^2
    # t0: log(10.1/9.9)^2 = 0.000407
    # t1: log(10.3/10.1)^2 = 0.000388
    # t2: log(10.2/10.0)^2 = 0.000392
    # beta(roll=2)
    # t1: 0.000407 + 0.000388 = 0.000795
    # t2: 0.000388 + 0.000392 = 0.000780
    # beta(roll=2 mean)
    # t2: (0.000795 + 0.000780) / 2 = 0.0007875
    assert np.isclose(beta.iloc[2], 0.0007875, atol=1e-5)
    assert beta.iloc[3] > 0
    assert beta.iloc[4] > 0

    # 2. Gamma
    gamma = gamma_estimates(high, low)
    # H_max(roll=2)
    # t1: max(10.1, 10.3) = 10.3
    # L_min(roll=2)
    # t1: min(9.9, 10.1) = 9.9
    # gamma[t1] = log(10.3/9.9)^2 = 0.00158
    assert np.isclose(gamma.iloc[1], 0.00158, atol=1e-5)

    # 3. Alpha
    alpha = alpha_estimates(beta, gamma)
    assert alpha.iloc[2] >= 0 # Should be floored at 0
    
    # 4. Spread
    spread = corwin_schultz_estimator(high, low, window)
    assert not spread.isna().all()
    assert spread.iloc[-1] >= 0

def test_bekker_parkinson(sample_hl_prices):
    """Test the Bekker-Parkinson estimator."""
    high, low = sample_hl_prices
    window = 2
    
    vol = bekker_parkinson_volatility_estimates(high, low, window)
    assert not vol.isna().all()
    assert vol.iloc[-1] >= 0