import sklearn.metrics as Metrics
import sklearn.ensemble as Ensemble
import sklearn.tree as Tree
import sklearn.datasets as Datasets
import sklearn.model_selection as ModelSelection
import pandas as pd
import numpy as np

"""
    function: Implementation of MDA
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 82 Mean-Decrease Accuracy section snippet 6.3 (snippet 8.3 2018)
"""
def feature_importance_MDA(
    classifier,  # classifier for fit and prediction
    X: pd.DataFrame,  # features matrix
    y: pd.DataFrame,  # labels vector
    n_splits: int,  # cross-validation n folds
    score_sample_weights: list = None,  # sample weights for score step
    train_sample_weights: list = None,  # sample weights for train step
) -> pd.DataFrame:

    train_sample_weights = np.ones(X.shape[0]) if train_sample_weights == None else train_sample_weights
    score_sample_weights = np.ones(X.shape[0]) if score_sample_weights == None else score_sample_weights

    cv_generator = ModelSelection.KFold(n_splits=n_splits)
    score0, score1 = pd.Series(), pd.DataFrame(columns=X.columns)

    # for each fold of k-fold
    for (i, (train, test)) in enumerate(cv_generator.split(X=X)):
        print(f"fold {i} start ...")

        X0, y0, sample_weights0 = X.iloc[train, :], y.iloc[train], train_sample_weights[train]
        X1, y1, sample_weights1 = X.iloc[test, :], y.iloc[test], score_sample_weights[test]

        fit = classifier.fit(X=X0, y=y0, sample_weight=sample_weights0)

        prediction_probability = fit.predict_proba(X1)  # prediction before shuffling
        score0.loc[i] = -Metrics.log_loss(
            y1,
            prediction_probability,
            labels=classifier.classes_,
            sample_weight=sample_weights1
        )

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)
            prediction_probability = fit.predict_proba(X1_)
            log_loss = Metrics.log_loss(
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
