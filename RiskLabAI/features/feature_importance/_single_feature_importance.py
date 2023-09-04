import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

def feature_importance_sfi(
    classifier: object,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    n_splits: int,
    score_sample_weights: list = None,
    train_sample_weights: list = None,
    scoring: str = "log_loss"
) -> pd.DataFrame:
    """
    Compute feature importance using the Single Feature Importance (SFI) method.

    :param classifier: Classifier for fit and prediction.
    :type classifier: object
    :param features: Features matrix.
    :type features: pd.DataFrame
    :param labels: Labels vector.
    :type labels: pd.DataFrame
    :param n_splits: Number of cross-validation folds.
    :type n_splits: int
    :param score_sample_weights: Sample weights for score step, default is None.
    :type score_sample_weights: list
    :param train_sample_weights: Sample weights for train step, default is None.
    :type train_sample_weights: list
    :param scoring: Scoring type for classification prediction and true values, "log_loss" or "accuracy", default is "log_loss".
    :type scoring: str
    :return: DataFrame containing feature importances.
    :rtype: pd.DataFrame
    """
    if train_sample_weights is None:
        train_sample_weights = np.ones(features.shape[0])
    if score_sample_weights is None:
        score_sample_weights = np.ones(features.shape[0])

    cv_generator = KFold(n_splits=n_splits)

    feature_names = features.columns
    importances = pd.DataFrame(columns=["FeatureName", "Mean", "StandardDeviation"])

    for feature_name in feature_names:
        scores = []

        for train, test in cv_generator.split(features):
            feature_train, label_train, sample_weights_train = (
                features.loc[train, [feature_name]], labels.iloc[train], train_sample_weights[train]
            )

            feature_test, label_test, sample_weights_test = (
                features.loc[test, [feature_name]], labels.iloc[test], score_sample_weights[test]
            )

            classifier.fit(feature_train, label_train, sample_weight=sample_weights_train)

            if scoring == "log_loss":
                prediction_probability = classifier.predict_proba(feature_test)
                score = -log_loss(label_test, prediction_probability, sample_weight=sample_weights_test, labels=classifier.classes_)

            elif scoring == "accuracy":
                prediction = classifier.predict(feature_test)
                score = accuracy_score(label_test, prediction, sample_weight=sample_weights_test)

            else:
                raise ValueError(f"'{scoring}' method not defined.")

            scores.append(score)

        importances = importances.append({
            "FeatureName": feature_name,
            "Mean": np.mean(scores),
            "StandardDeviation": np.std(scores, ddof=1) * len(scores) ** -0.5
        }, ignore_index=True)

    return importances
