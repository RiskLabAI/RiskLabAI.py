import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List

def group_mean_std(
    dataframe: pd.DataFrame,
    clusters: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate the group mean and standard deviation for the given clusters.

    This function aggregates the values of the columns in the dataframe that
    belong to each cluster and calculates their mean and standard deviation.

    :param dataframe: Dataframe containing feature importances.
    :param clusters: Dictionary of feature clusters.
    :return: Dataframe containing the mean and standard deviation for each cluster.

    Reference:
        De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS, page 86, "Clustered MDI" section.
    """
    output = pd.DataFrame(columns=["Mean", "StandardDeviation"])

    for cluster_index, column_indices in clusters.items():
        cluster_data = dataframe[column_indices].sum(axis=1)
        output.loc["C_" + str(cluster_index), "Mean"] = cluster_data.mean()
        output.loc["C_" + str(cluster_index), "StandardDeviation"] = \
            cluster_data.std() * cluster_data.shape[0] ** -0.5

    return output

def clustered_feature_importance_mdi(
    classifier: RandomForestClassifier,
    feature_names: List[str],
    clusters: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate clustered feature importance using Mean Decrease Impurity (MDI).

    This function calculates the clustered feature importance by aggregating the
    feature importances from individual trees in a random forest classifier.

    :param classifier: The classifier (Random Forest) to be used.
    :param feature_names: List of feature names.
    :param clusters: Dictionary of feature clusters.
    :return: Dataframe containing feature importance of each cluster.

    Reference:
        De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS, page 86, "Clustered MDI" section.
    """
    importances_dict = {i: tree.feature_importances_ for i, tree in enumerate(classifier.estimators_)}
    importances_df = pd.DataFrame.from_dict(importances_dict, orient="index")
    importances_df.columns = feature_names
    importances_df = importances_df.replace(0, np.nan)  # because max_features=1

    aggregated_importances = group_mean_std(importances_df, clusters)
    aggregated_importances /= aggregated_importances["Mean"].sum()

    return aggregated_importances
