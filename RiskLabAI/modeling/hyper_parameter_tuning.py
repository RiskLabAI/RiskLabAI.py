import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline

class MyPipeline(Pipeline):
    """
    Custom pipeline class to include sample_weight in fit_params.
    """
    def fit(self, X, y, sample_weight=None, **fit_params):
        """
        Fit the pipeline while considering sample weights.
        
        :param X: Feature data.
        :type X: pd.DataFrame
        :param y: Labels of data.
        :type y: pd.DataFrame
        :param sample_weight: Sample weights for fit, defaults to None.
        :type sample_weight: list or None
        :param **fit_params: Additional fit parameters.
        :return: Fitted pipeline.
        :rtype: MyPipeline
        """
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)

def clf_hyper_fit(
        feature_data,  # Data of features
        label,  # Labels of data
        time,  # Observation time
        pipe_clf,  # Our estimator
        param_grid,  # Parameter space
        cv=3,  # Number of groups for cross validation
        bagging=[0, -1, 1.],  # Bagging type
        rnd_search_iter=0,
        n_jobs=-1,
        percent_embargo=0,  # Percent of embargo
        **fit_params
):
    """
    Perform hyperparameter tuning and model fitting.

    :param feature_data: Data of features.
    :type feature_data: pd.DataFrame
    :param label: Labels of data.
    :type label: pd.DataFrame
    :param time: Observation time.
    :type time: float
    :param pipe_clf: Our estimator.
    :type pipe_clf: sklearn.pipeline.Pipeline
    :param param_grid: Parameter space.
    :type param_grid: dict
    :param cv: Number of groups for cross validation, defaults to 3.
    :type cv: int
    :param bagging: Bagging type, defaults to [0, -1, 1.].
    :type bagging: list
    :param rnd_search_iter: Number of iterations for randomized search, defaults to 0.
    :type rnd_search_iter: int
    :param n_jobs: Number of jobs for parallel processing, defaults to -1.
    :type n_jobs: int
    :param percent_embargo: Percent of embargo, defaults to 0.
    :type percent_embargo: int
    :param **fit_params: Additional fit parameters.
    :return: Fitted pipeline.
    :rtype: MyPipeline
    """
    if set(label.values) == {0, 1}:
        scoring = 'f1'  # F1-score for meta-labeling
    else:
        scoring = 'neg_log_loss'  # Symmetric towards all cases

    # Hyperparameter search on train data
    inner_cv = PurgedKFold(n_splits=cv, times=time, percent_embargo=percent_embargo)  # Purged
    if rnd_search_iter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring,
                                cv=inner_cv, n_jobs=n_jobs, n_iter=rnd_search_iter)
    gs = gs.fit(feature_data, label, **fit_params).best_estimator_  # Pipeline

    # Fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]), max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feature_data, label, sample_weight=fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])

    return gs
