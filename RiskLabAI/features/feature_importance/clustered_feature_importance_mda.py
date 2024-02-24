from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
# from feature_importance_strategy import FeatureImportanceStrategy

from RiskLabAI.features.feature_importance.feature_importance_strategy import FeatureImportanceStrategy
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


class ClusteredFeatureImportanceMDA(FeatureImportanceStrategy):

    def __init__():
        pass

    def compute(
            self,
            classifier: RandomForestClassifier,
            x: pd.DataFrame,
            y: pd.Series,
            clusters: Dict[str, List[str]],
            n_splits: int = 10,
            score_sample_weights: List[float] = None,
            train_sample_weights: List[float] = None
    ) -> pd.DataFrame:
        """
        Compute clustered feature importance using MDA.

        The feature importance is computed by comparing the performance
        (log loss) of a trained classifier on shuffled data to its 
        performance on non-shuffled data.

        :param classifier: The Random Forest classifier to be trained.
        :param x: The features DataFrame.
        :param y: The target Series.
        :param clusters: A dictionary where the keys are the cluster names
                         and the values are lists of features in each cluster.
        :param n_splits: The number of splits for KFold cross-validation.
        :param score_sample_weights: Sample weights to be used when computing the score.
        :param train_sample_weights: Sample weights to be used during training.
        
        :return: A DataFrame with feature importances and their standard deviations.

        The related mathematical formulae:
        
        .. math::

            \\text{{importance}} = \\frac{{-1 \\times \\text{{score with shuffled data}}}}
                                  {{\\text{{score without shuffled data}}}}
        
        Using Central Limit Theorem for calculating the standard deviation:

        .. math::

            \\text{{StandardDeviation}} = \\text{{std}} \\times n^{-0.5}

        """

        # Handle default values for sample weights
        if train_sample_weights is None:
            train_sample_weights = np.ones(x.shape[0])
        if score_sample_weights is None:
            score_sample_weights = np.ones(x.shape[0])

        cv_generator = KFold(n_splits=n_splits)
        baseline_scores, shuffled_scores = pd.Series(dtype=float), pd.DataFrame(columns=clusters.keys())

        for i, (train, test) in enumerate(cv_generator.split(X=x)):
            print(f"Fold {i} start ...")

            x_train, y_train, train_weights = x.iloc[train, :], y.iloc[train], train_sample_weights[train]
            x_test, y_test, test_weights = x.iloc[test, :], y.iloc[test], score_sample_weights[test]

            classifier_fit = classifier.fit(X=x_train, y=y_train, sample_weight=train_weights)
            prediction_probability = classifier_fit.predict_proba(x_test)

            baseline_scores[i] = -log_loss(y_test, prediction_probability, labels=classifier.classes_, sample_weight=test_weights)

            for cluster_name in shuffled_scores.columns:
                x_test_shuffled = x_test.copy(deep=True)
                for feature in clusters[cluster_name]:
                    np.random.shuffle(x_test_shuffled[feature].values)
                prob = classifier_fit.predict_proba(x_test_shuffled)
                shuffled_scores.loc[i, cluster_name] = -log_loss(y_test, prob, labels=classifier.classes_)

        importances = (-1 * shuffled_scores).add(baseline_scores, axis=0)
        importances /= -1 * shuffled_scores

        # Central Limit Theorem for standard deviation
        importances = pd.concat(
            {"Mean": importances.mean(), "StandardDeviation": importances.std() * (importances.shape[0] ** -0.5)}, axis=1
        )

        importances.index = ["C_" + str(i) for i in importances.index]
        return importances
