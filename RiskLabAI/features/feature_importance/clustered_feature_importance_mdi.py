# from feature_importance_strategy import FeatureImportanceStrategy
from RiskLabAI.features.feature_importance.feature_importance_strategy import FeatureImportanceStrategy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List


class ClusteredFeatureImportanceMDI(FeatureImportanceStrategy):

    def __init__(
            self,
            classifier: RandomForestClassifier,
            clusters: Dict[str, List[str]],
            x: pd.DataFrame,
            y: pd.Series
    ):
        """
        Initialize the ClusteredFeatureImportanceMDI class.

        :param classifier: The Random Forest classifier.
        :param clusters: A dictionary where the keys are the cluster names 
                         and the values are lists of features in each cluster.
        :param x: The features DataFrame.
        :param y: The target Series.
        """
        self.classifier = classifier
        self.clusters = clusters
        classifier.fit(x, y)

    def group_mean_std(
            self,
            dataframe: pd.DataFrame,
            clusters: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Calculate the mean and standard deviation for clusters.

        :param dataframe: A DataFrame of importances.
        :param clusters: A dictionary of cluster definitions.
        
        :return: A DataFrame with mean and standard deviation for each cluster.

        Using Central Limit Theorem for standard deviation:

        .. math::

            \\text{{StandardDeviation}} = \\text{{std}} \\times n^{-0.5}

        """
        output = pd.DataFrame(columns=["Mean", "StandardDeviation"])

        for cluster_name, feature_names in clusters.items():
            cluster_data = dataframe[feature_names].sum(axis=1)
            output.loc["C_" + str(cluster_name), "Mean"] = cluster_data.mean()
            output.loc["C_" + str(cluster_name), "StandardDeviation"] = (
                cluster_data.std() * (cluster_data.shape[0] ** -0.5)
            )

        return output

    def compute(self) -> pd.DataFrame:
        """
        Compute aggregated feature importances for clusters.

        :return: A DataFrame with aggregated importances for clusters.
        """
        importances_dict = {i: tree.feature_importances_ for i, tree in enumerate(self.classifier.estimators_)}
        importances_df = pd.DataFrame.from_dict(importances_dict, orient="index")
        importances_df.columns = self.classifier.feature_names_in_
        importances_df.replace(0, np.nan, inplace=True)  # because max_features=1

        aggregated_importances = self.group_mean_std(importances_df, self.clusters)
        aggregated_importances /= aggregated_importances["Mean"].sum()

        return aggregated_importances
