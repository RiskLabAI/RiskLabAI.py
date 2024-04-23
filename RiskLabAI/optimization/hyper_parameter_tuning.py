import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline

from RiskLabAI.backtest.validation.cross_validator_controller import CrossValidatorController


class MyPipeline(Pipeline):
    """
    Custom pipeline class to include sample_weight in fit_params.
    """

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame, 
        sample_weight: list = None, 
        **fit_params
    ) -> 'MyPipeline':
        """
        Fit the pipeline while considering sample weights.
        
        :param X: Feature data.
        :param y: Labels of data.
        :param sample_weight: Sample weights for fit, defaults to None.
        :param **fit_params: Additional fit parameters.
        :return: Fitted pipeline.
        """
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


def clf_hyper_fit(
    feature_data: pd.DataFrame,
    label: pd.DataFrame,
    times: pd.Series,
    pipe_clf: Pipeline,
    param_grid: dict,
    validator_type: str = 'purgedkfold',
    validator_params: dict = None,
    bagging: list = [0, -1, 1.],
    rnd_search_iter: int = 0,
    n_jobs: int = -1,
    **fit_params
) -> MyPipeline:
    """
    Perform hyperparameter tuning and model fitting.

    :param feature_data: Data of features.
    :param label: Labels of data.
    :param times: The timestamp series associated with the labels.
    :param pipe_clf: Our estimator.
    :param param_grid: Parameter space.
    :param validator_type: Type of cross-validator to create.
    :param validator_params: Additional keyword arguments to be passed to the cross-validator's constructor.
    :param bagging: Bagging type, defaults to [0, -1, 1.].
    :param rnd_search_iter: Number of iterations for randomized search, defaults to 0.
    :param n_jobs: Number of jobs for parallel processing, defaults to -1.
    :param **fit_params: Additional fit parameters.
    :return: Fitted pipeline.
    """
    if set(label.values) == {0, 1}:
        scoring = 'f1'  # F1-score for meta-labeling
    else:
        scoring = 'neg_log_loss'  # Symmetric towards all cases

    if validator_params is None:
        validator_params = {
            'times' : times,
            'n_splits' : 5,
            'embargo' : 0.01,
        }

    # Hyperparameter search on train data
    inner_cv =  CrossValidatorController(
        validator_type,
        **validator_params
    ).cross_validator
    
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
