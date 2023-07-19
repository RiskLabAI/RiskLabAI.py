import sklearn.metrics as Metrics
import sklearn.ensemble as Ensemble
import sklearn.tree as Tree
import sklearn.datasets as Datasets
import sklearn.model_selection as ModelSelection
import pandas as pd
import numpy as np

"""
function: Clustered feature importance MDA
reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
methodology: page 87 Clustered MDA section
"""
def clustered_feature_importance_MDA(
    classifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
    clusters: dict,
    n_splits: int = 10,
    score_sample_weights: list = None,  # sample weights for score step
    train_sample_weights: list = None,  # sample weights for train step
) -> pd.DataFrame:

    train_sample_weights = np.ones(X.shape[0]) if train_sample_weights == None else train_sample_weights
    score_sample_weights = np.ones(X.shape[0]) if score_sample_weights == None else score_sample_weights

    cv_generator = ModelSelection.KFold(n_splits=n_splits)
    score0, score1 = pd.Series(), pd.DataFrame(columns=clusters.keys())
    for i, (train, test) in enumerate(cv_generator.split(X=X)):
        print(f"fold {i} start ...")

        X0, y0, sample_weights0 = X.iloc[train, :], y.iloc[train], train_sample_weights[train]
        X1, y1, sample_weights1 = X.iloc[test, :], y.iloc[test], score_sample_weights[test]

        fit = classifier.fit(X=X0, y=y0, sample_weight=sample_weights0)

        prediction_probability = fit.predict_proba(X1)
        score0.loc[i] = -Metrics.log_loss(
            y1,
            prediction_probability,
            labels=classifier.classes_,
            sample_weight=sample_weights1
        )

        for j in score1.columns:
            X1_ = X1.copy(deep=True)
            for k in clusters[j]:
                np.random.shuffle(X1_[k].values)  # shuffle cluster
                prob = fit.predict_proba(X1_)
                score1.loc[i, j] = -Metrics.log_loss(y1, prob, labels=classifier.classes_)

    importances = (-1 * score1).add(score0, axis=0)
    importances = importances / (-1 * score1)

    importances = pd.concat({
        "Mean": importances.mean(),
        "StandardDeviation": importances.std()*importances.shape[0]**-0.5
    }, axis=1)  # CLT

    importances.index = ["C_"+str(i) for i in importances.index]
    return importances
