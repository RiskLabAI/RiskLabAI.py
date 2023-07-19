# using PyCall
# using Statistics
# using PlotlyJS
# using TimeSeries
# using Random
# using Distributions

import numpy as np
import pandas as pd
import sklearn.datasets as Datasets

"""
function: Generating a set of informed, redundant and explanatory variables
reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
methodology: page 77 A Few Caveats of p-Values section snippet 6.1 (snippet 8.7 2018)
"""


def get_test_dataset(
    n_features: int = 100,  # total number of features
    n_informative: int = 25,  # number of informative features
    n_redundant: int = 25,  # number of redundant features
    n_samples: int = 10000,  # number of sample to generate
    random_state: int = 0,  # random state
    sigma_std: float = 0.0,  # standard deviation of generation
) -> tuple:
    # generate a random dataset for a classiﬁcation problem
    np.random.seed(random_state)

    X, y = Datasets.make_classification(
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


# # generate a random dataset for a classification problem
# from sklearn.datasets import make_classification
# np.random.seed(random_state)
# X,y=make_classification(n_samples=n_samples,
# n_features=n_features-n_redundant,
# n_informative=n_informative,n_redundant=0,shuffle=False,
# random_state=random_state)
# cols=[‘I_’+str(i) for i in xrange(n_informative)]
# cols+=[‘N_’+str(i) for i in xrange(n_features-n_informative- \
# n_redundant)]
# X,y=pd.DataFrame(X,columns=cols),pd.Series(y)
# i=np.random.choice(xrange(n_informative),size=n_redundant)
# for k,j in enumerate(i):
# X[‘R_’+str(k)]=X[‘I_’+str(j)]+np.random.normal(size= \
# X.shape[0])*sigmaStd
# return X,y
