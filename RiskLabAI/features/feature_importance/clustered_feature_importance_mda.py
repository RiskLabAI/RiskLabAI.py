"""
Computes Clustered Mean Decrease Accuracy (MDA) feature importance.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from .feature_importance_strategy import FeatureImportanceStrategy

class ClusteredFeatureImportanceMDA(FeatureImportanceStrategy):
    """
    Computes clustered feature importance using MDA.

    This method shuffles entire *clusters* of features at a time
    and measures the decrease in model performance.
    """


    def __init__(
        self,
        classifier: object,
        clusters: Dict[str, List[str]],
        n_splits: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : object
            An *untrained* scikit-learn classifier.
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster names to lists of feature names.
        n_splits : int, default=10
            Number of splits for cross-validation.
        random_state : int, default=42
            Seed for KFold and shuffling for reproducibility.
        """
        self.classifier = classifier
        self.clusters = clusters
        self.n_splits = n_splits
        self.random_state = random_state


    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute Clustered MDA feature importance.

        Parameters
        ----------
        x : pd.DataFrame
            The feature data.
        y : pd.Series
            The target data.
        **kwargs : Any
            - 'train_sample_weights': Optional sample weights for training.
            - 'score_sample_weights': Optional sample weights for scoring.

        Returns
        -------
        pd.DataFrame
            DataFrame with "Mean" and "StandardDeviation" of importance
            for each *cluster*.
        """
        train_weights = kwargs.get('train_sample_weights')
        score_weights = kwargs.get('score_sample_weights')

        if train_weights is None:
            train_weights = np.ones(x.shape[0])
        if score_weights is None:
            score_weights = np.ones(x.shape[0])

        cv_generator = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        baseline_scores = pd.Series(dtype=float)
        shuffled_scores = pd.DataFrame(columns=self.clusters.keys(), dtype=float)   

        for i, (train_idx, test_idx) in enumerate(cv_generator.split(X=x)):
            print(f"Fold {i} start ...")

            x_train, y_train, w_train = (
                x.iloc[train_idx, :],
                y.iloc[train_idx],
                train_weights[train_idx],
            )
            x_test, y_test, w_test = (
                x.iloc[test_idx, :],
                y.iloc[test_idx],
                score_weights[test_idx],
            )

            classifier_fit = self.classifier.fit(
                X=x_train, y=y_train, sample_weight=w_train
            )
            prediction_probability = classifier_fit.predict_proba(x_test)

            baseline_scores.loc[i] = -log_loss(
                y_test,
                prediction_probability,
                labels=self.classifier.classes_,
                sample_weight=w_test,
            )

            # Get scores for each shuffled *cluster*
            rng = np.random.default_rng(self.random_state + i)
            for cluster_name in shuffled_scores.columns:
                x_test_shuffled = x_test.copy(deep=True)
                
                # --- CORRECTED SHUFFLING LOGIC ---
                # Get all feature names for this cluster
                cluster_cols = self.clusters[cluster_name]
                
                if not cluster_cols: # Skip if cluster is empty
                    shuffled_scores.loc[i, cluster_name] = baseline_scores.loc[i]
                    continue
                    
                # Get the underlying numpy array for these columns
                cluster_data = x_test_shuffled[cluster_cols].values
                
                # Shuffle the rows of this array in-place.
                # This applies the *same* permutation to all features
                # in the cluster, preserving intra-cluster correlation.
                rng.shuffle(cluster_data)
                
                # Assign the shuffled data back
                x_test_shuffled[cluster_cols] = cluster_data
                # --- END CORRECTION ---
                
                prob = classifier_fit.predict_proba(x_test_shuffled)
                shuffled_scores.loc[i, cluster_name] = -log_loss(
                    y_test, prob, labels=self.classifier.classes_,
                    sample_weight=w_test  
                )

        # Calculate importance as the simple drop in score
        importances = shuffled_scores.rsub(baseline_scores, axis=0)

        # Central Limit Theorem for standard deviation
        importances_summary = pd.concat(
            {
                "Mean": importances.mean(),
                "StandardDeviation": (
                    importances.std() * (importances.shape[0] ** -0.5)
                ),
            },
            axis=1,
        )

        importances_summary.index = [
            f"C_{i}" for i in importances_summary.index
        ]
        return importances_summary