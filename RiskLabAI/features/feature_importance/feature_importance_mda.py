"""
Computes Mean Decrease Accuracy (MDA) feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from typing import List, Optional, Any, Callable
from .feature_importance_strategy import FeatureImportanceStrategy

class FeatureImportanceMDA(FeatureImportanceStrategy):
    """
    Computes feature importance using Mean Decrease Accuracy (MDA).

    This method shuffles each feature one by one and measures how
    much the model's performance (e.g., log loss) decreases.
    """


    def __init__(self, classifier: object, n_splits: int = 10, random_state: int = 42):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : object
            An *untrained* scikit-learn classifier.
        n_splits : int, default=10
            Number of splits for cross-validation.
        """
        self.classifier = classifier
        self.n_splits = n_splits
        self.random_state = random_state


    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute MDA feature importance.

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
            DataFrame with "Mean" and "StandardDeviation" of importance.
        """
        train_weights = kwargs.get('train_sample_weights')
        score_weights = kwargs.get('score_sample_weights')
        
        if train_weights is None:
            train_weights = np.ones(x.shape[0])
        if score_weights is None:
            score_weights = np.ones(x.shape[0])


        cv_generator = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        baseline_scores = pd.Series(dtype=float)
        shuffled_scores = pd.DataFrame(columns=x.columns, dtype=float)

        for i, (train_idx, test_idx) in enumerate(cv_generator.split(x)):
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

            # Fit classifier and get baseline score
            fitted_classifier = self.classifier.fit(
                X=x_train, y=y_train, sample_weight=w_train
            )
            pred_proba = fitted_classifier.predict_proba(x_test)

            baseline_scores.loc[i] = -log_loss(
                y_test,
                pred_proba,
                labels=self.classifier.classes_,
                sample_weight=w_test,
            )

            # Get scores for each shuffled feature
            rng = np.random.default_rng(self.random_state + i) 
            for feature in x.columns:
                x_test_shuffled = x_test.copy(deep=True)
                rng.shuffle(x_test_shuffled[feature].values) # <-- USE SEEDED SHUFFLE

                shuffled_proba = fitted_classifier.predict_proba(x_test_shuffled)
                
                shuffled_scores.loc[i, feature] = -log_loss(
                    y_test,
                    shuffled_proba,
                    labels=self.classifier.classes_,
                    sample_weight=w_test
                )

        # Calculate importance as the simple drop in score
        importances = shuffled_scores.rsub(baseline_scores, axis=0)
        
        # Calculate mean and std dev
        importances_summary = pd.concat(
            {
                "Mean": importances.mean(),
                "StandardDeviation": (
                    importances.std() * (importances.shape[0] ** -0.5)
                ),
            },
            axis=1,
        )

        return importances_summary