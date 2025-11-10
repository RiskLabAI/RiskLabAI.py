"""
Tests for CrossValidatorController.
"""

import pandas as pd
import pytest


from RiskLabAI.backtest.validation.combinatorial_purged import CombinatorialPurged
from RiskLabAI.backtest.validation.cross_validator_controller import CrossValidatorController
from RiskLabAI.backtest.validation.kfold import KFold

@pytest.fixture
def dummy_args():
    """Dummy 'times' args."""
    idx = pd.date_range('2020-01-01', periods=10)
    times = pd.Series(idx, index=idx)
    return {
        "n_splits": 5,
        "n_test_groups": 2,
        "times": times,
    }

def test_controller_creates_kfold():
    """Test controller with KFold."""
    controller = CrossValidatorController(
        validator_type='kfold', n_splits=10
    )
    validator = controller.get_validator()
    
    assert isinstance(validator, KFold)
    assert validator.n_splits == 10

def test_controller_creates_cpcv(dummy_args):
    """Test controller with CombinatorialPurged."""
    controller = CrossValidatorController(
        validator_type='combinatorialpurged',
        **dummy_args
    )
    validator = controller.get_validator()
    
    assert isinstance(validator, CombinatorialPurged)
    assert validator.n_splits == 5
    assert validator.n_test_groups == 2

def test_controller_public_attribute(dummy_args):
    """Test accessing the validator as a public attribute."""
    controller = CrossValidatorController(
        validator_type='combinatorialpurged',
        **dummy_args
    )
    assert isinstance(controller.cross_validator, CombinatorialPurged)