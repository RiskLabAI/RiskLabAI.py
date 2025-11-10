"""
Computes Mean Decrease Impurity (MDI) feature importance.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Any
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble import BaseEnsemble  
from .feature_importance_strategy import FeatureImportanceStrategy

class FeatureImportanceMDI(FeatureImportanceStrategy):
    """
    Computes feature importance using Mean Decrease Impurity (MDI).

    This method is specific to tree-based ensembles (like RandomForest)
    and measures importance as the average impurity decrease.
    """

    def __init__(self, classifier: BaseEnsemble):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : BaseEnsemble
            An *untrained* scikit-learn ensemble model (e.g., RandomForestClassifier).
        """
        if not isinstance(classifier, BaseEnsemble):
            raise TypeError("Classifier must be an ensemble (e.g., RandomForest).")

        self.classifier = classifier

    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute MDI feature importance.

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
            DataFrame with "Mean" and "StandardDeviation" of importance.
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
        
        # Ensure correct feature names
        if hasattr(self.classifier, 'feature_names_in_'):
             importances_df.columns = self.classifier.feature_names_in_
        else:
             importances_df.columns = x.columns

        # Replace 0 with NaN (as per user's original code)
        importances_df.replace(0, np.nan, inplace=True)

        importances = pd.concat(
            {
                "Mean": importances_df.mean(),
                "StandardDeviation": (
                    importances_df.std()
                    * (importances_df.shape[0] ** -0.5)
                ),
            },
            axis=1,
        )

        # Normalize
        importances /= importances["Mean"].sum()
        return importances