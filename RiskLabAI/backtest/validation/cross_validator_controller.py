from .cross_validator_factory import CrossValidatorFactory

class CrossValidatorController:
    """
    Controller class to handle the cross-validation process.
    """

    def __init__(
        self,
        validator_type: str,
        **kwargs
    ):
        """
        Initializes the CrossValidatorController.

        :param validator_type: Type of cross-validator to create and use.
            This is passed to the factory to instantiate the appropriate cross-validator.
        :type validator_type: str

        :param kwargs: Additional keyword arguments to be passed to the cross-validator's constructor.
        :type kwargs: Type
        """
        self.cross_validator = CrossValidatorFactory.create_cross_validator(
            validator_type,
            **kwargs
        )

"""
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.linear_model import LogisticRegression

# Sample data (for illustration purposes only)
n_samples = 120
times_range = pd.date_range(start='01-01-2020', periods=n_samples, freq='D')

data = pd.DataFrame({
    'Feature1': np.random.randn(n_samples),
    'Feature2': np.random.randn(n_samples),
    'Target': np.random.choice([0, 1], size=n_samples),
}, index=times_range)

X = data[['Feature1', 'Feature2']]
y = data['Target']
times = pd.Series(times_range + pd.Timedelta('1d'), index=times_range)

cv = CrossValidatorController(
    validator_type='combinatorialpurged',
    n_splits=6,
    n_test_groups=2,
    times=times,
    embargo=0.01,
).cross_validator

for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {idx + 1}:")
    print("Train Indices:", train_idx)
    print("Test Indices:", test_idx)
    print("-----")

paths = cv.backtest_paths(X)

for key, value in paths.items():
    paths[key] = [dict(subdict) for subdict in value]  # Convert each numpy record to a regular dictionary
    for subdict in paths[key]:
        subdict['Train'] = list(subdict['Train'])
        subdict['Test'] = list(subdict['Test'])

pprint(paths)

cv.backtest_predictions(LogisticRegression(), X, y)
"""
