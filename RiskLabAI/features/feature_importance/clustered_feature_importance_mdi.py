from feature_importance_strategy import FeatureImportanceStrategy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List


class ClusteredFeatureImportanceMDI(FeatureImportanceStrategy):
    def __init__(self, classifier: RandomForestClassifier, clusters: Dict[str, List[str]], x, y):
        self.classifier = classifier
        self.clusters = clusters
        classifier.fit(x, y)

    def group_mean_std(self, dataframe: pd.DataFrame, clusters: Dict[str, List[str]]) -> pd.DataFrame:
        output = pd.DataFrame(columns=["Mean", "StandardDeviation"])

        for cluster_index, column_indices in clusters.items():
            cluster_data = dataframe[column_indices].sum(axis=1)
            output.loc["C_" + str(cluster_index), "Mean"] = cluster_data.mean()
            output.loc["C_" + str(cluster_index), "StandardDeviation"] = (
                cluster_data.std() * cluster_data.shape[0] ** -0.5
            )

        return output

    def compute(self) -> pd.DataFrame:
        importances_dict = {i: tree.feature_importances_ for i, tree in enumerate(self.classifier.estimators_)}
        importances_df = pd.DataFrame.from_dict(importances_dict, orient="index")
        importances_df.columns = self.classifier.feature_names_in_
        importances_df = importances_df.replace(0, np.nan)  # because max_features=1

        aggregated_importances = self.group_mean_std(importances_df, self.clusters)
        aggregated_importances /= aggregated_importances["Mean"].sum()

        return aggregated_importances
