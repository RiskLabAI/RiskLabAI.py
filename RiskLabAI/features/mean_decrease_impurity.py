import pandas as pd
import numpy as np


def feature_importance_mdi(
        classifier: object,
        feature_names: list
) -> pd.DataFrame:
    """
    Calculate feature importances using Mean-Decrease Impurity (MDI) method.

    This method calculates the importance of a feature based on the number of times
    a feature is used to split the data across all trees.

    :param classifier: Classifier for fit and prediction
    :type classifier: object
    :param feature_names: List of feature names
    :type feature_names: list
    :return: Dataframe containing feature importances
    :rtype: pd.DataFrame
    """
    feature_importances_dict = {i: tree.feature_importances_ for i, tree in enumerate(classifier.estimators_)}

    feature_importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient="index")
    feature_importances_df.columns = feature_names

    feature_importances_df = feature_importances_df.replace(0, np.nan)  # because max_features=1

    importances = pd.concat({
        "Mean": feature_importances_df.mean(),
        "StandardDeviation": feature_importances_df.std() * feature_importances_df.shape[0]**-0.5
    }, axis=1)  # CLT

    importances /= importances["Mean"].sum()
    return importances
