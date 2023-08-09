import sklearn.metrics as metrics
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
import pandas as pd
import numpy as np

def clustered_feature_importance_MDA(
    classifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
    clusters: dict,
    n_splits: int = 10,
    score_sample_weights: list = None,
    train_sample_weights: list = None,
) -> pd.DataFrame:
    """
    Calculate clustered feature importance using Mean Decrease Accuracy (MDA).

    This function calculates the clustered feature importance by measuring the change
    in model accuracy when the features in each cluster are shuffled. This method helps
    in understanding the importance of feature clusters in relation to the model accuracy.

    :param classifier: The classifier to be used.
    :param X: Feature data.
    :param y: Target data.
    :param clusters: Dictionary of feature clusters.
    :param n_splits: Number of splits for cross-validation. Default is 10.
    :param score_sample_weights: Sample weights for score step. Default is None.
    :param train_sample_weights: Sample weights for train step. Default is None.
    :return: Dataframe containing feature importance of each cluster.

    Reference:
        De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS, page 87, "Clustered MDA" section.
    """
    if train_sample_weights is None:
        train_sample_weights = np.ones(X.shape[0])
    if score_sample_weights is None:
        score_sample_weights = np.ones(X.shape[0])

    cv_generator = model_selection.KFold(n_splits=n_splits)
    score0, score1 = pd.Series(dtype=float), pd.DataFrame(columns=clusters.keys())

    for i, (train, test) in enumerate(cv_generator.split(X=X)):
        print(f"Fold {i} start ...")

        X0, y0, weights0 = X.iloc[train, :], y.iloc[train], train_sample_weights[train]
        X1, y1, weights1 = X.iloc[test, :], y.iloc[test], score_sample_weights[test]

        fit = classifier.fit(X=X0, y=y0, sample_weight=weights0)
        prediction_probability = fit.predict_proba(X1)

        score0[i] = -metrics.log_loss(
            y1,
            prediction_probability,
            labels=classifier.classes_,
            sample_weight=weights1
        )

        for j in score1.columns:
            X1_ = X1.copy(deep=True)
            for k in clusters[j]:
                np.random.shuffle(X1_[k].values)
                prob = fit.predict_proba(X1_)
                score1.loc[i, j] = -metrics.log_loss(y1, prob, labels=classifier.classes_)

    importances = (-1 * score1).add(score0, axis=0)
    importances /= (-1 * score1)

    importances = pd.concat({
        "Mean": importances.mean(),
        "StandardDeviation": importances.std() * importances.shape[0] ** -0.5
    }, axis=1)  # Central Limit Theorem

    importances.index = ["C_" + str(i) for i in importances.index]
    return importances
