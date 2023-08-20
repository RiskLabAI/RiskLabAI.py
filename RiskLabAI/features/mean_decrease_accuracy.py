import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


def feature_importance_mda(
        classifier: object,
        x: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
        score_sample_weights: list = None,
        train_sample_weights: list = None
) -> pd.DataFrame:
    """
    Calculate feature importances using Mean-Decrease Accuracy (MDA) method.

    :param classifier: Classifier for fit and prediction
    :type classifier: object
    :param x: Features matrix
    :type x: pd.DataFrame
    :param y: Labels vector
    :type y: pd.Series
    :param n_splits: Number of cross-validation folds
    :type n_splits: int
    :param score_sample_weights: Sample weights for score step, default is None
    :type score_sample_weights: list, optional
    :param train_sample_weights: Sample weights for train step, default is None
    :type train_sample_weights: list, optional
    :return: Dataframe containing feature importances
    :rtype: pd.DataFrame
    """
    if train_sample_weights is None:
        train_sample_weights = np.ones(x.shape[0])
    if score_sample_weights is None:
        score_sample_weights = np.ones(x.shape[0])

    cv_generator = KFold(n_splits=n_splits)
    score0, score1 = pd.Series(dtype=float), pd.DataFrame(columns=x.columns)

    for i, (train, test) in enumerate(cv_generator.split(x)):
        print(f"fold {i} start ...")

        x0, y0, sample_weights0 = x.iloc[train, :], y.iloc[train], train_sample_weights[train]
        x1, y1, sample_weights1 = x.iloc[test, :], y.iloc[test], score_sample_weights[test]

        fit = classifier.fit(x=x0, y=y0, sample_weight=sample_weights0)
        prediction_probability = fit.predict_proba(x1)
        score0.loc[i] = -log_loss(
            y1,
            prediction_probability,
            labels=classifier.classes_,
            sample_weight=sample_weights1
        )

        for j in x.columns:
            x1_ = x1.copy()
            np.random.shuffle(x1_[j].values)
            prediction_probability = fit.predict_proba(x1_)
            score1.loc[i, j] = -log_loss(
                y1,
                prediction_probability,
                labels=classifier.classes_
            )

    importances = (-1 * score1).add(score0, axis=0)
    importances /= -1 * score1

    importances = pd.concat({
        "Mean": importances.mean(),
        "StandardDeviation": importances.std() * importances.shape[0]**-0.5
    }, axis=1)  # CLT

    return importances
