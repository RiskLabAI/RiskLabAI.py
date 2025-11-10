"""
Tests for CrossValidatorFactory.
"""

import numpy as np
import pandas as pd
import pytest

# Import all validator classes
from RiskLabAI.backtest.validation.adaptive_combinatorial_purged import AdaptiveCombinatorialPurged
from RiskLabAI.backtest.validation.bagged_combinatorial_purged import BaggedCombinatorialPurged
from RiskLabAI.backtest.validation.combinatorial_purged import CombinatorialPurged
from RiskLabAI.backtest.validation.cross_validator_factory import CrossValidatorFactory
from RiskLabAI.backtest.validation.kfold import KFold
from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold
from RiskLabAI.backtest.validation.test_walk_forward import WalkForward

@pytest.fixture
def dummy_args():
    """Dummy 'times' and 'feature' args for complex validators."""
    idx = pd.date_range('2020-01-01', periods=10)
    times = pd.Series(idx, index=idx)
    feature = pd.Series(np.arange(10), index=idx)
    return {
        "n_splits": 5,
        "n_test_groups": 2,
        "times": times,
        "external_feature": feature
    }

def test_factory_kfold():
    cv = CrossValidatorFactory.create_cross_validator('kfold', n_splits=5)
    assert isinstance(cv, KFold)
    assert cv.n_splits == 5

def test_factory_walkforward():
    cv = CrossValidatorFactory.create_cross_validator('walkforward', n_splits=3, gap=1)
    assert isinstance(cv, WalkForward)
    assert cv.gap == 1

def test_factory_purgedkfold(dummy_args):
    cv = CrossValidatorFactory.create_cross_validator(
        'purgedkfold', n_splits=5, times=dummy_args['times']
    )
    assert isinstance(cv, PurgedKFold)

def test_factory_combinatorialpurged(dummy_args):
    cv = CrossValidatorFactory.create_cross_validator(
        'combinatorialpurged', **dummy_args
    )
    assert isinstance(cv, CombinatorialPurged)

def test_factory_bagged(dummy_args):
    cv = CrossValidatorFactory.create_cross_validator(
        'baggedcombinatorialpurged', **dummy_args
    )
    assert isinstance(cv, BaggedCombinatorialPurged)

def test_factory_adaptive(dummy_args):
    cv = CrossValidatorFactory.create_cross_validator(
        'adaptivecombinatorialpurged', **dummy_args
    )
    assert isinstance(cv, AdaptiveCombinatorialPurged)

def test_factory_case_insensitivity(dummy_args):
    """Test that the factory is case-insensitive."""
    cv = CrossValidatorFactory.create_cross_validator(
        'PurgedKFold', n_splits=5, times=dummy_args['times']
    )
    assert isinstance(cv, PurgedKFold)

def test_factory_invalid_type():
    """Test that an invalid type raises a ValueError."""
    with pytest.raises(ValueError) as exc_info:
        CrossValidatorFactory.create_cross_validator('invalid_type')
    assert "Invalid validator_type: invalid_type" in str(exc_info.value)