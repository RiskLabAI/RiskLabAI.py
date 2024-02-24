# from feature_importance_strategy import FeatureImportanceStrategy
from RiskLabAI.features.feature_importance.feature_importance_strategy import FeatureImportanceStrategy

import pandas as pd
import numpy as np
from typing import List, Optional, Union


class FeatureImportanceMDI(FeatureImportanceStrategy):
    """
    Computes the feature importance using the Mean Decrease Impurity (MDI) method.

    The method calculates the importance of a feature by measuring the average impurity
    decrease across all the trees in the forest, where impurity is calculated 
    using metrics like Gini impurity or entropy.

    .. math::

        \\text{importance}_{j} = \\frac{\\text{average impurity decrease for feature j}}{\\text{total impurity decrease}}

    """

    def __init__(
            self,
            classifier: object,
            x: pd.DataFrame,
            y: Union[pd.Series, List[Optional[float]]]
    ) -> None:
        """
        Initialize the class with parameters.

        :param classifier: The classifier object.
        :param x: The feature data.
        :param y: The target data.
        """
        self.classifier = classifier
        classifier.fit(x, y)

    def compute(self) -> pd.DataFrame:
        """
        Compute the feature importances.

        :return: Feature importances as a dataframe with "Mean" and "StandardDeviation" columns.
        """
        feature_importances_dict = {i: tree.feature_importances_ for i, tree in enumerate(self.classifier.estimators_)}
        feature_importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient="index")
        feature_importances_df.columns = self.classifier.feature_names_in_

        # Replace 0 with NaN to avoid inaccuracies in calculations
        feature_importances_df.replace(0, np.nan, inplace=True)  

        importances = pd.concat({
            "Mean": feature_importances_df.mean(),
            "StandardDeviation": feature_importances_df.std() * (feature_importances_df.shape[0] ** -0.5)
        }, axis=1)

        # Normalize importances to sum up to 1
        importances /= importances["Mean"].sum()

        return importances
