import numpy as np
import pandas as pd
import sklearn.datasets as datasets

def get_test_dataset(
    n_features: int = 100,
    n_informative: int = 25,
    n_redundant: int = 25,
    n_samples: int = 10000,
    random_state: int = 0,
    sigma_std: float = 0.0
) -> tuple:
    """
    Generate a synthetic dataset with informative, redundant, and explanatory variables.

    :param n_features: Total number of features
    :type n_features: int
    :param n_informative: Number of informative features
    :type n_informative: int
    :param n_redundant: Number of redundant features
    :type n_redundant: int
    :param n_samples: Number of samples to generate
    :type n_samples: int
    :param random_state: Random state for reproducibility
    :type random_state: int
    :param sigma_std: Standard deviation for generating redundant features, default is 0.0
    :type sigma_std: float
    :return: Tuple containing generated X (features) and y (labels)
    :rtype: tuple
    """
    np.random.seed(random_state)

    X, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features - n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state,
    )

    columns = ["I_" + str(i) for i in range(n_informative)]
    columns += ["N_" + str(i) for i in range(n_features - n_informative - n_redundant)]
    X, y = pd.DataFrame(X, columns=columns), pd.Series(y)

    i = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X["R_" + str(k)] = X["I_" + str(j)] + np.random.normal(size=X.shape[0]) * sigma_std

    return X, y
