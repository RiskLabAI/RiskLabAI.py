import pandas as pd
import numpy as np
from statistics import mean, stdev
import sklearn.metrics as Metrics
import sklearn.model_selection as ModelSelection

def feature_importance_SFI(
    classifier, 
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    n_splits: int, 
    score_sample_weights: list = None, 
    train_sample_weights: list = None, 
    scoring: str = "log_loss"
) -> pd.DataFrame:
    """
    Compute Feature Importance using the SFI method.

    :param classifier: Classifier for fit and prediction
    :type classifier: object
    :param X: Features matrix
    :type X: pd.DataFrame
    :param y: Labels vector
    :type y: pd.DataFrame
    :param n_splits: Number of cross-validation folds
    :type n_splits: int
    :param score_sample_weights: Sample weights for score step, default is None
    :type score_sample_weights: list
    :param train_sample_weights: Sample weights for train step, default is None
    :type train_sample_weights: list
    :param scoring: Scoring type for classification prediction and true values, "log_loss" or "accuracy", default is "log_loss"
    :type scoring: str
    :return: DataFrame containing feature importances
    :rtype: pd.DataFrame
    """
    train_sample_weights = np.ones(X.shape[0]) if train_sample_weights is None else train_sample_weights
    score_sample_weights = np.ones(X.shape[0]) if score_sample_weights is None else score_sample_weights

    cv_generator = ModelSelection.KFold(n_splits=n_splits)

    feature_names = X.columns
    importances = pd.DataFrame(columns=["FeatureName", "Mean", "StandardDeviation"])
    for feature_name in feature_names:
        scores = []
        for (i, (train, test)) in enumerate(cv_generator.split(X)):

            X0, y0, sample_weights0 = X.loc[train, [feature_name]], y.iloc[train], train_sample_weights[train]
            X1, y1, sample_weights1 = X.loc[test, [feature_name]], y.iloc[test], score_sample_weights[test]

            fit = classifier.fit(X0, y0, sample_weight=sample_weights0)

            if scoring == "log_loss":
                prediction_probability = fit.predict_proba(X1)
                score_ = -Metrics.log_loss(y1, prediction_probability, sample_weight=sample_weights1, labels=classifier.classes_)

            elif scoring == "accuracy":
                prediction = fit.predict(X1)
                score_ = Metrics.accuracy_score(y1, prediction, sample_weight=sample_weights1)

            else:
                raise ValueError(f"'{scoring}' method not defined.")

            scores.append(score_)

        importances = importances.append({
            "FeatureName": feature_name, 
            "Mean": mean(scores), 
            "StandardDeviation": stdev(scores) * len(scores) ** -0.5
        },
        ignore_index=True)

    return importances
