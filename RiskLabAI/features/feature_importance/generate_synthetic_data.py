"""
Generates a synthetic dataset for feature importance testing.

Based on the method from De Prado (2018).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Tuple

def get_test_dataset(
    n_features: int = 100,
    n_informative: int = 25,
    n_redundant: int = 25,
    n_samples: int = 10000,
    random_state: int = 0,
    sigma_std: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate a synthetic dataset with informative, redundant, and noise features.

    Parameters
    ----------
    n_features : int, default=100
        Total number of features to generate.
    n_informative : int, default=25
        Number of informative features.
    n_redundant : int, default=25
        Number of redundant features. These are created by adding
        Gaussian noise to informative features.
    n_samples : int, default=10000
        Number of samples (rows) to generate.
    random_state : int, default=0
        Random state for reproducibility.
    sigma_std : float, default=0.0
        Standard deviation of the noise added to create redundant features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        - X (pd.DataFrame): The generated feature matrix.
        - y (pd.Series): The generated labels.
    """
    rng = np.random.default_rng(random_state)

    # 1. Generate informative and noise features
    n_noise = n_features - n_informative - n_redundant
    x_array, y_array = make_classification(
        n_samples=n_samples,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state,
    )

    # 2. Create DataFrame
    columns = [f"I_{i}" for i in range(n_informative)]
    columns += [f"N_{i}" for i in range(n_noise)]
    x_df = pd.DataFrame(x_array, columns=columns)
    y_series = pd.Series(y_array, name="Target")

    # 3. Create redundant features
    # Randomly pick informative features to copy
    redundant_indices = rng.choice(
        range(n_informative), size=n_redundant, replace=True
    )
    
    for i, orig_idx in enumerate(redundant_indices):
        orig_feature_name = f"I_{orig_idx}"
        new_feature_name = f"R_{i}"
        
        # Add noise to the original informative feature
        noise = rng.normal(
            loc=0.0, scale=sigma_std, size=n_samples
        ) * x_df[orig_feature_name].std()
        
        x_df[new_feature_name] = x_df[orig_feature_name] + noise

    return x_df[sorted(x_df.columns)], y_series