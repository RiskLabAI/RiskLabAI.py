# feature_importance_factory.py

from feature_importance_strategy import (
    FeatureImportanceStrategy,
)  # Assuming the abstract class is in the same directory
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Union, Optional
import pandas as pd


class FeatureImportanceFactory:
    def __init__(self) -> FeatureImportanceStrategy:
        pass

    def build(
        self,
        feature_importance_strategy: FeatureImportanceStrategy,
    ) -> FeatureImportanceStrategy:
        self.results = feature_importance_strategy.compute()
        return self

    def get(
        self,
    ) -> pd.DataFrame:
        return self.results
