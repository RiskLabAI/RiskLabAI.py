import sklearn.datasets as datasets
import sklearn.ensemble as ensemble
import pandas as pd
import numpy as np

def feature_importance_MDI(
    classifier, 
    feature_names: list
) -> pd.DataFrame:
    """
    Calculate feature importances using Mean-Decrease Impurity (MDI) method.

    :param classifier: Classifier for fit and prediction
    :type classifier: object
    :param feature_names: List of feature names
    :type feature_names: list
    :return: Dataframe containing feature importances
    :rtype: pd.DataFrame
    """
    dict0 = {i: tree.feature_importances_ for i, tree in enumerate(classifier.estimators_)}

    dataframe0 = pd.DataFrame.from_dict(dict0, orient="index")
    dataframe0.columns = feature_names

    dataframe0 = dataframe0.replace(0, np.nan)  # because max_features=1

    importances = pd.concat({
        "Mean": dataframe0.mean(),
        "StandardDeviation": dataframe0.std() * dataframe0.shape[0] ** -0.5
    }, axis=1)  # CLT

    importances /= importances["Mean"].sum()
    return importances
