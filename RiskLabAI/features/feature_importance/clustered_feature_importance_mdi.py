"""
Computes Clustered Mean Decrease Impurity (MDI) feature importance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import BaseEnsemble
from typing import Dict, List, Any
from .feature_importance_strategy import FeatureImportanceStrategy

class ClusteredFeatureImportanceMDI(FeatureImportanceStrategy):
    """
    Computes Clustered MDI feature importance.

    Aggregates MDI importance for pre-defined clusters of features.
    """

    def __init__(
        self,
        classifier: BaseEnsemble,
        clusters: Dict[str, List[str]],
    ):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : BaseEnsemble
            An *untrained* scikit-learn ensemble model.
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster names to lists of feature names.
        """
        if not isinstance(classifier, BaseEnsemble):
            raise TypeError("Classifier must be an ensemble (e.g., RandomForest).")

        self.classifier = classifier
        self.clusters = clusters

    def _group_mean_std(
        self, dataframe: pd.DataFrame, clusters: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Calculate the mean and standard deviation for cluster importances.
        """
        output = pd.DataFrame(columns=["Mean", "StandardDeviation"])

        for cluster_name, feature_names in clusters.items():
            # Sum importance for all features in the cluster
            cluster_data = dataframe[feature_names].sum(axis=1)
            
            cluster_mean = cluster_data.mean()
            cluster_std = cluster_data.std()
            
            output.loc[f"C_{cluster_name}", "Mean"] = cluster_mean
            output.loc[f"C_{cluster_name}", "StandardDeviation"] = (
                cluster_std * (cluster_data.shape[0] ** -0.5)
            )
        return output

    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute Clustered MDI feature importance.

        Parameters
        ----------
        x : pd.DataFrame
            The feature data.
        y : pd.Series
            The target data.
        **kwargs : Any
            Keyword arguments for the classifier's `fit` method
            (e.g., `sample_weight`).

        Returns
        -------
        pd.DataFrame
            DataFrame with "Mean" and "StandardDeviation" of importance
            for each *cluster*.
        """
        train_sample_weights = kwargs.get('sample_weight')
        
        # Fit the classifier
        self.classifier.fit(x, y, sample_weight=train_sample_weights)
        
        # Get importance from each tree
        importances_dict = {
            i: tree.feature_importances_
            for i, tree in enumerate(self.classifier.estimators_)
        }
        importances_df = pd.DataFrame.from_dict(importances_dict, orient="index")
        
        if hasattr(self.classifier, 'feature_names_in_'):
             importances_df.columns = self.classifier.feature_names_in_
        else:
             importances_df.columns = x.columns

        # Replace 0 with NaN
        importances_df.replace(0, np.nan, inplace=True)

        # Group by cluster
        aggregated_importances = self._group_mean_std(importances_df, self.clusters)
        
        # Normalize
        aggregated_importances /= aggregated_importances["Mean"].sum()
        return aggregated_importances