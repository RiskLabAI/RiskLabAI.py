from feature_importance_strategy import FeatureImportanceStrategy  
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

class FeatureImportanceMDA(FeatureImportanceStrategy):
    def __init__(
        self,
        classifier: object,
        x: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 10,
        score_sample_weights: list = None,
        train_sample_weights: list = None
    ):
        self.classifier = classifier
        self.x = x
        self.y = y
        self.n_splits = n_splits
        self.score_sample_weights = score_sample_weights
        self.train_sample_weights = train_sample_weights

    def compute(self) -> pd.DataFrame:
        if self.train_sample_weights is None:
            self.train_sample_weights = np.ones(self.x.shape[0])
        if self.score_sample_weights is None:
            self.score_sample_weights = np.ones(self.x.shape[0])

        cv_generator = KFold(n_splits=self.n_splits)
        score0, score1 = pd.Series(dtype=float), pd.DataFrame(columns=self.x.columns)

        for i, (train, test) in enumerate(cv_generator.split(self.x)):
            print(f"Fold {i} start ...")

            x0, y0, weights0 = self.x.iloc[train, :], self.y.iloc[train], self.train_sample_weights[train]
            x1, y1, weights1 = self.x.iloc[test, :], self.y.iloc[test], self.score_sample_weights[test]

            fit = self.classifier.fit(X=x0, y=y0, sample_weight=weights0)
            prediction_probability = fit.predict_proba(x1)

            score0.loc[i] = -log_loss(
                y1,
                prediction_probability,
                labels=self.classifier.classes_,
                sample_weight=weights1
            )

            for j in self.x.columns:
                x1_ = x1.copy(deep=True)
                np.random.shuffle(x1_[j].values)
                prob = fit.predict_proba(x1_)
                score1.loc[i, j] = -log_loss(y1, prob, labels=self.classifier.classes_)

        importances = (-1 * score1).add(score0, axis=0)
        importances /= (-1 * score1)

        importances = pd.concat({
            "Mean": importances.mean(),
            "StandardDeviation": importances.std() * importances.shape[0]**-0.5
        }, axis=1)  # CLT

        return importances
