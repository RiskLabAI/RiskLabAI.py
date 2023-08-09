import sklearn.metrics as metrics
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
import pandas as pd
import numpy as np

def feature_importance_MDA(
    classifier, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int, 
    score_sample_weights: list = None, 
    train_sample_weights: list = None
) -> pd.DataFrame:
    """
    Calculate feature importances using Mean-Decrease Accuracy (MDA) method.

    :param classifier: Classifier for fit and prediction
    :type classifier: object
    :param X: Features matrix
    :type X: pd.DataFrame
    :param y: Labels vector
    :type y: pd.Series
    :param n_splits: Number of cross-validation folds
    :type n_splits: int
    :param score_sample_weights: Sample weights for score step, default is None
    :type score_sample_weights: list
    :param train_sample_weights: Sample weights for train step, default is None
    :type train_sample_weights: list
    :return: Dataframe containing feature importances
    :rtype: pd.DataFrame
    """
    train_sample_weights = np.ones(X.shape[0]) if train_sample_weights is None else train_sample_weights
    score_sample_weights = np.ones(X.shape[0]) if score_sample_weights is None else score_sample_weights

    cv_generator = model_selection.KFold(n_splits=n_splits)
    score0, score1 = pd.Series(), pd.DataFrame(columns=X.columns)

    for (i, (train, test)) in enumerate(cv_generator.split(X=X)):
        print(f"fold {i} start ...")

        X0, y0, sample_weights0 = X.iloc[train, :], y.iloc[train], train_sample_weights[train]
        X1, y1, sample_weights1 = X.iloc[test, :], y.iloc[test], score_sample_weights[test]

        fit = classifier.fit(X=X0, y=y0, sample_weight=sample_weights0)

        prediction_probability = fit.predict_proba(X1)
        score0.loc[i] = -metrics.log_loss(
            y1,
            prediction_probability,
            labels=classifier.classes_,
            sample_weight=sample_weights1
        )

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)
            prediction_probability = fit.predict_proba(X1_)
            log_loss = metrics.log_loss(
                y1,
                prediction_probability,
                labels=classifier.classes_)

            score1.loc[i, j] = -log_loss

    importances = (-1 * score1).add(score0, axis=0)
    importances = importances / (-1 * score1)

    importances = pd.concat({
        "Mean": importances.mean(),
        "StandardDeviation": importances.std()*importances.shape[0]**-0.5
    }, axis=1)  # CLT

    return importances
