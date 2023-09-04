from feature_importance_strategy import FeatureImportanceStrategy  # Assuming the abstract class is in the same directory
import pandas as pd
import numpy as np

class FeatureImportanceMDI(FeatureImportanceStrategy):
    def __init__(self, classifier: object, feature_names: list):
        self.classifier = classifier
        self.feature_names = feature_names

    def compute(self) -> pd.DataFrame:
        feature_importances_dict = {i: tree.feature_importances_ for i, tree in enumerate(self.classifier.estimators_)}
        feature_importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient='index')
        feature_importances_df.columns = self.feature_names

        feature_importances_df = feature_importances_df.replace(0, np.nan)  # because max_features=1

        importances = pd.concat({
            "Mean": feature_importances_df.mean(),
            "StandardDeviation": feature_importances_df.std() * feature_importances_df.shape[0]**-0.5
        }, axis=1)  # CLT

        importances /= importances["Mean"].sum()

        return importances
