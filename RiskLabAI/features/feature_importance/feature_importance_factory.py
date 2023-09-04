# feature_importance_factory.py

from feature_importance_strategy import FeatureImportanceStrategy  # Assuming the abstract class is in the same directory
from feature_importance_mdi import FeatureImportanceMDI
from feature_importance_mda import FeatureImportanceMDA
from feature_importance_clustered_mdi import FeatureImportanceClusteredMDI
from feature_importance_clustered_mda import FeatureImportanceClusteredMDA  # Add this import
from feature_importance_sfi import FeatureImportanceSFI
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Union, Optional
import pandas as pd

class FeatureImportanceFactory:
    def create_strategy(
        self,
        strategy_type: str,
        classifier: RandomForestClassifier,
        features: pd.DataFrame,
        labels: pd.Series,
        n_splits: int = 5,
        score_sample_weights: Optional[List[float]] = None,
        train_sample_weights: Optional[List[float]] = None,
        feature_names: Optional[List[str]] = None,
        clusters: Optional[Dict[str, List[str]]] = None,
        scoring: Optional[str] = "log_loss"
    ) -> FeatureImportanceStrategy:

        if strategy_type == 'MDI':
            return FeatureImportanceMDI(classifier, feature_names)

        elif strategy_type == 'ClusteredMDI':
            return FeatureImportanceClusteredMDI(classifier, feature_names, clusters)

        elif strategy_type == 'MDA':
            return FeatureImportanceMDA(classifier, features, labels, n_splits, score_sample_weights, train_sample_weights)

        elif strategy_type == 'ClusteredMDA':  # Add this condition
            return FeatureImportanceClusteredMDA(classifier, features, labels, n_splits, score_sample_weights, train_sample_weights, clusters)

        elif strategy_type == 'SFI':
            return FeatureImportanceSFI(classifier, features, labels, n_splits, score_sample_weights, train_sample_weights, scoring)

        else:
            raise ValueError(f"Invalid strategy_type: {strategy_type}")
