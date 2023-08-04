from statistics import mean, stdev
import sklearn.metrics as Metrics
import sklearn.ensemble as Ensemble

import sklearn.datasets as Datasets
import sklearn.metrics as Metrics
import sklearn.model_selection as ModelSelection

import pandas as pd
import numpy as np

"""
    function: Implementation of SFI method
    reference: De Prado, M. (2018) Advances In Financial Machine Learning
    methodology: page 118 SFI section snippet 8.4
"""
def feature_importance_SFI(
    classifier,  # classifier for fit and prediction
    X: pd.DataFrame,  # features matrix
    y: pd.DataFrame,  # labels vector
    n_splits: int,  # cross-validation n folds
    score_sample_weights: list = None,  # sample weights for score step
    train_sample_weights: list = None,  # sample weights for train step
    scoring:str="log_loss" # classification prediction and true values scoring type 
) -> pd.DataFrame:

    train_sample_weights = np.ones(X.shape[0]) if train_sample_weights == None else train_sample_weights
    score_sample_weights = np.ones(X.shape[0]) if score_sample_weights == None else score_sample_weights

    cvGenerator = ModelSelection.KFold(n_splits=n_splits)

    feature_names = X.columns
    importances = pd.DataFrame(columns=["FeatureName", "Mean", "StandardDeviation"])
    for feature_name in feature_names:
        scores = []
        for (i, (train, test)) in enumerate(cvGenerator.split(X)):
    
            X0, y0, sample_weights0 = X.loc[train, [feature_name]], y.iloc[train], train_sample_weights[train]
            X1, y1, sample_weights1 = X.loc[test, [feature_name]], y.iloc[test], score_sample_weights[test]

            
            fit = classifier.fit(X0, y0, sample_weight=sample_weights0)

            if scoring == "log_loss":
                predictionProbability = fit.predict_proba(X1)
                score_ = -Metrics.log_loss(y1, predictionProbability, sample_weight=sample_weights1 ,labels=classifier.classes_)        
            
            elif scoring == "accuracy":
                prediction = fit.predict(X1)
                score_ = Metrics.accuracy_score(y1, prediction, sample_weight=sample_weights1)
            
            else:
                raise(f"'{scoring}' method not defined.")

            scores.append(score_)

        importances = importances.append({
            "FeatureName":feature_name, "Mean":mean(scores), "StandardDeviation":stdev(scores) * len(scores) ** -0.5},
            ignore_index = True
        )

    return importances

