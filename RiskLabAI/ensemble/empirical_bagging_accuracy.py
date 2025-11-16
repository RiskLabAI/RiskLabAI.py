"""
Implements the BaggingClassifierAccuracy class to evaluate the
performance of a bagging classifier using various weighting schemes,
as described in "Advances in Financial Machine Learning" by de Prado (2018),
Chapter 6, Section 6.5, pp. 91-92.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from scipy.stats import norm
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns


class BaggingClassifierAccuracy:
    """
    Evaluates a bagging classifier's accuracy using different
    weighting schemes based on decision tree c_i scores.

    Methods:
    - fit: Fits the bagging classifier.
    - calculate_c_i: Calculates the c_i score for each tree.
    - calculate_weights: Computes weights (uniform, c_i, 1-c_i^2).
    - predict: Predicts class labels using specified weights.
    - evaluate_all_schemes: Gets accuracy for all weighting schemes.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        max_samples: int = 100,
        max_features: float = 1.0,
        random_state: Optional[int] = None
    ):
        """
        Initializes the BaggingClassifier.

        Note: This class uses a specific base_estimator:
        DecisionTreeClassifier(criterion='entropy', max_features=1,
        class_weight='balanced') as per the book's context.

        Parameters
        ----------
        n_estimators : int, default=1000
            Number of trees in the ensemble.
        max_samples : int, default=100
            Number of samples to draw for training each tree.
        max_features : float, default=1.0
            Number of features to draw for training each tree.
        random_state : int, optional
            Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

        self.base_estimator = DecisionTreeClassifier(
            criterion='entropy',
            max_features=1,  # Trees vote on one feature
            class_weight='balanced'
        )
        
        self.clf = BaggingClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        self.estimators_ = None
        self.weights_ = None
        self.c_i_scores_ = None
        # <-- ADDED: Store class labels
        self.class_0_ = None
        self.class_1_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaggingClassifierAccuracy':
        """
        Fits the bagging classifier on the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training labels.

        Returns
        -------
        self
            Fitted object.
        """
        self.clf.fit(X, y)
        self.estimators_ = self.clf.estimators_
        
        # <-- ADDED: Check for binary classification and store classes
        if len(self.clf.classes_) != 2:
            raise ValueError("This class only supports binary classification.")
            
        self.class_0_ = self.clf.classes_[0]
        self.class_1_ = self.clf.classes_[1]
        
        return self

    def calculate_c_i(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Calculates the c_i score (accuracy) for each decision tree
        on the full training set.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training labels.

        Returns
        -------
        np.ndarray
            Array of c_i scores for each estimator.
        """
        if self.estimators_ is None:
            raise NotFittedError("Classifier must be fitted first. Call .fit()")
            
        c_i_scores = []
        for tree in self.estimators_:
            y_pred = tree.predict(X)
            acc = accuracy_score(y, y_pred)
            c_i_scores.append(acc)
            
        self.c_i_scores_ = np.array(c_i_scores)
        return self.c_i_scores_

    def calculate_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, np.ndarray]:
        """
        Calculates weights for each estimator based on three schemes:
        1. Uniform (w_i = 1/N)
        2. c_i (w_i = c_i / sum(c_i))
        3. 1 - c_i^2 (w_i = (1 - c_i^2) / sum(1 - c_i^2))

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training labels.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of weight arrays for each scheme.
        """
        if self.c_i_scores_ is None:
            # calculate_c_i also checks if model is fitted
            self.calculate_c_i(X, y)
            
        c_i = self.c_i_scores_
        n = len(c_i)

        # 1. Uniform weights
        w_uniform = np.ones(n) / n

        # 2. c_i weights
        sum_c_i = np.sum(c_i)
        w_c_i = c_i / sum_c_i if sum_c_i != 0 else w_uniform

        # 3. 1 - c_i^2 weights (proportional to variance)
        c_i_squared = c_i**2
        w_variance = 1. - c_i_squared
        sum_w_var = np.sum(w_variance)
        w_variance = w_variance / sum_w_var if sum_w_var != 0 else w_uniform
        
        self.weights_ = {
            'uniform': w_uniform,
            'c_i': w_c_i,
            'variance': w_variance
        }
        return self.weights_

    def predict(
        self,
        X: pd.DataFrame,
        weight_scheme: str = 'uniform'
    ) -> np.ndarray:
        """
        Predicts class labels for X using the specified weighting scheme.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.
        weight_scheme : str, default='uniform'
            The weighting scheme to use ('uniform', 'c_i', 'variance').

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        # <-- UPDATED: Check fit status first
        if self.estimators_ is None:
            raise NotFittedError("Classifier must be fitted first. Call .fit()")
            
        if self.weights_ is None:
            # <-- UPDATED: More specific error
            raise NotFittedError("Weights must be calculated first. Call .calculate_weights()")
            
        if weight_scheme not in self.weights_:
            raise ValueError(f"Unknown weight_scheme: {weight_scheme}. "
                             f"Must be one of {list(self.weights_.keys())}")

        weights = self.weights_[weight_scheme]
        
        # Get predictions from each tree
        # (N_samples, N_estimators)
        tree_preds = np.array([tree.predict(X) for tree in self.estimators_]).T
        
        # <-- UPDATED: Convert labels {class_0, class_1} to {-1, 1}
        # Map class_1 to 1, and class_0 to -1
        tree_preds_signed = np.where(tree_preds == self.class_1_, 1, -1)
        
        # Calculate weighted average vote
        # (N_samples, N_estimators) * (N_estimators,) -> (N_samples,)
        weighted_votes = np.dot(tree_preds_signed, weights)
        
        # <-- UPDATED: Convert vote back to {class_0, class_1}
        # Positive vote -> class_1, Negative or Zero vote -> class_0
        y_pred = np.where(weighted_votes > 0, self.class_1_, self.class_0_)
        
        return y_pred

    def evaluate_all_schemes(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, float]:
        """
        Fits, calculates weights, and evaluates accuracy for all
        three weighting schemes.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test labels.
        X_train : pd.DataFrame
            Training features (for calculating weights).
        y_train : pd.Series
            Training labels (for calculating weights).

        Returns
        -------
        Dict[str, float]
            Dictionary of accuracy scores for each scheme.
        """
        # Fit classifier
        self.fit(X_train, y_train)
        
        # Calculate weights (uses X_train, y_train implicitly)
        self.calculate_weights(X_train, y_train)
        
        accuracies = {}
        for scheme in self.weights_.keys():
            y_pred = self.predict(X_test, weight_scheme=scheme)
            acc = accuracy_score(y_test, y_pred)
            accuracies[scheme] = acc
            
        return accuracies


# --- Standalone Functions for Bootstrap Analysis ---

def calculate_bootstrap_accuracy(
    clf: BaggingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstraps: int = 1000
) -> Tuple[np.ndarray, float, float]:
    """
    Calculates the accuracy of a bagging classifier over multiple
    bootstrapped samples of the test set.

    This helps estimate the standard deviation of the accuracy (std(a_n)).

    Parameters
    ----------
    clf : BaggingClassifier
        A fitted BaggingClassifier instance.
    X : pd.DataFrame
        Test features.
    y : pd.Series
        Test labels.
    n_bootstraps : int, default=1000
        Number of bootstrap iterations.

    Returns
    -------
    Tuple[np.ndarray, float, float]
        - a_n_values: Array of accuracy scores from each bootstrap.
        - a_n_mean: The mean accuracy.
        - a_n_std: The standard deviation of the accuracies.
    """
    a_n_values = []
    n_samples = len(y)
    
    # Use indices from the original X/y DataFrames/Series
    indices = X.index
    
    # --- CHANGE: Fixed typo n_bootstraMps -> n_bootstraps ---
    for _ in range(n_bootstraps):
    # --- END CHANGE ---
        # Sample test set with replacement
        boot_indices = np.random.choice(indices, n_samples, replace=True)
        X_boot = X.loc[boot_indices]
        y_boot = y.loc[boot_indices]
        
        # Predict on the bootstrapped sample
        y_pred = clf.predict(X_boot)
        acc = accuracy_score(y_boot, y_pred)
        a_n_values.append(acc)
        
    a_n_values = np.array(a_n_values)
    a_n_mean = np.mean(a_n_values)
    a_n_std = np.std(a_n_values, ddof=1)
    
    return a_n_values, a_n_mean, a_n_std


def plot_bootstrap_accuracy_distribution(
    a_n_values: np.ndarray,
    a_n_mean: float,
    a_n_std: float,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the distribution of bootstrapped accuracy scores.

    Parameters
    ----------
    a_n_values : np.ndarray
        Array of accuracy scores from each bootstrap.
    a_n_mean : float
        The mean accuracy.
    a_n_std : float
        The standard deviation of the accuracies.
    ax : plt.Axes, optional
        Matplotlib Axes object to plot on. If None, creates a new one.

    Returns
    -------
    plt.Axes
        The Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(a_n_values, kde=True, ax=ax, stat='density', label='Empirical Distribution')
    
    # Overlay a normal distribution
    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max, 100)
    p = norm.pdf(x, a_n_mean, a_n_std)
    ax.plot(x, p, 'k', linewidth=2, label=f'Normal(μ={a_n_mean:.3f}, σ={a_n_std:.3f})')
    
    ax.legend()
    return ax