"""
Hyperparameter tuning module that integrates with scikit-learn
and the custom PurgedKFold cross-validators.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Optional, Union

# Import the controller from the refactored validation module
from RiskLabAI.backtest.validation import CrossValidatorController


class MyPipeline(Pipeline):
    """
    Custom pipeline class to correctly pass `sample_weight` to the
    final estimator's `fit` method.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        **fit_params,
    ) -> "MyPipeline":
        """
        Fit the pipeline, passing `sample_weight` to the final step.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        y : pd.Series
            Labels.
        sample_weight : np.ndarray, optional
            Sample weights.
        **fit_params : Any
            Additional fit parameters.

        Returns
        -------
        MyPipeline
            The fitted pipeline.
        """
        if sample_weight is not None:
            # Add sample_weight to fit_params for the *last step*
            step_name = self.steps[-1][0]
            fit_params[f"{step_name}__sample_weight"] = sample_weight
            
        return super().fit(X, y, **fit_params)


def clf_hyper_fit(
    feature_data: pd.DataFrame,
    label: pd.Series,
    times: pd.Series,
    pipe_clf: Pipeline,
    param_grid: Dict[str, Any],
    validator_type: str = "purgedkfold",
    validator_params: Optional[Dict[str, Any]] = None,
    bagging: Optional[List[Union[int, float]]] = None,
    rnd_search_iter: int = 0,
    n_jobs: int = -1,
    **fit_params,
) -> Union[GridSearchCV, RandomizedSearchCV, Pipeline]:
    """
    Perform hyperparameter tuning using a specified cross-validator.

    Parameters
    ----------
    feature_data : pd.DataFrame
        Feature data (X).
    label : pd.Series
        Labels (y).
    times : pd.Series
        Series of event start and end times for purging.
    pipe_clf : Pipeline
        The scikit-learn pipeline (or `MyPipeline`) to tune.
    param_grid : dict
        Parameter grid for the search.
    validator_type : str, default='purgedkfold'
        The key for the cross-validator to use
        (e.g., 'purgedkfold', 'combinatorialpurged').
    validator_params : dict, optional
        Parameters for the cross-validator (e.g., `n_splits`, `embargo`).
    bagging : List, optional
        If provided, wraps the best estimator in a `BaggingClassifier`.
        Format: [n_estimators, max_samples, max_features].
        Default [0, -1, 1.0] from original code seems incorrect,
        so changed to None.
    rnd_search_iter : int, default=0
        If 0, use `GridSearchCV`.
        If > 0, use `RandomizedSearchCV` with this many iterations.
    n_jobs : int, default=-1
        Number of parallel jobs for the search.
    **fit_params : Any
        Additional fit parameters (e.g., `sample_weight`).

    Returns
    -------
    Union[GridSearchCV, RandomizedSearchCV, Pipeline]
        The fitted grid search object, or a fitted Bagging pipeline.
    """
    if bagging is None:
        bagging = [0, 0.0, 1.0] # Default to no bagging

    if set(label.unique()) == {0, 1}:
        scoring = "f1"  # F1-score for meta-labeling
    else:
        scoring = "neg_log_loss"

    if validator_params is None:
        validator_params = {
            "times": times,
            "n_splits": 5,
            "embargo": 0.01,
        }
    else:
        # Ensure 'times' is passed if not already present
        if 'times' not in validator_params:
            validator_params['times'] = times

    # 1. Set up the custom cross-validator
    inner_cv = CrossValidatorController(
        validator_type, **validator_params
    ).cross_validator

    # 2. Set up the hyperparameter search
    if rnd_search_iter == 0:
        gs = GridSearchCV(
            estimator=pipe_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
        )
    else:
        gs = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            n_iter=rnd_search_iter,
        )
        
    # 3. Fit the search
    gs = gs.fit(feature_data, label, **fit_params)

    # 4. (Optional) Fit bagging classifier on the best model
    if bagging[0] > 0:
        best_estimator = gs.best_estimator_
        
        # Create a new pipeline with the best estimator's steps
        bag_pipe = MyPipeline(best_estimator.steps)
        
        bag_clf = BaggingClassifier(
            estimator=bag_pipe,
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]) if bagging[1] > 0 else 1.0,
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
        )
        
        # Fit the bagging classifier
        bag_clf = bag_clf.fit(feature_data, label, **fit_params)
        
        # Return as a pipeline
        return Pipeline([("bag", bag_clf)])
    
    # 5. Return the best estimator found
    return gs.best_estimator_