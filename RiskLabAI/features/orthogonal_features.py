from ctypes import Union
import pandas as pd
import numpy as np

"""
    function: Implementation of Orthogonal Features (Compute Eigen Vectors)
    reference: De Prado, M. (2018) Advances In Financial Machine Learning
    methodology: page 119 Orthogonal Features section snippet 8.5
"""
def eigen_vectors(
    dot_product: np.ndarray,  # input dot product matrix
    explained_variance_threshold: float  # threshold for variance filtering
) -> pd.DataFrame:
    e_values, e_vectors = np.linalg.eigh(dot_product)
    
    eigen_dataframe = pd.DataFrame({
        "Index": [f"PC {i}" for i in range(1, len(e_values) + 1)],
        "EigenValue": e_values,
        "EigenVector": [ev for ev in e_vectors]
    })

    eigen_dataframe = eigen_dataframe.sort_values("EigenValue", ascending=False)

    cumulative_variance = np.cumsum(eigen_dataframe["EigenValue"]) / np.sum(eigen_dataframe["EigenValue"])

    eigen_dataframe["CumulativeVariance"] = cumulative_variance

    index = cumulative_variance.values.searchsorted(explained_variance_threshold)

    eigen_dataframe = eigen_dataframe.iloc[:index+1,:]

    return eigen_dataframe


"""
    function: Implementation of Orthogonal Features
    reference: De Prado, M. (2018) Advances In Financial Machine Learning
    methodology: page 119 Orthogonal Features section snippet 8.5
"""
def orthogonal_features(
    X:np.ndarray, # features matrix
    variance_threshold:float=0.95, # threshold for variance filtering
) -> tuple:

    # Given a dataframe X of features, compute orthofeatures P
    Z = (X - X.mean(axis=0)) / X.std(axis=0) # standardize
    dot_product = Z.T @ Z
    eigen_dataframe = eigen_vectors(dot_product, variance_threshold)

    W = np.vstack(eigen_dataframe["EigenVector"].values).T

    P = Z @ W

    return P, eigen_dataframe
