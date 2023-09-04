from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from FeatureImportanceStrategy import FeatureImportanceStrategy

class FeatureImportanceStrategy:
    def calculate_importance(self, *args, **kwargs) -> pd.DataFrame:
        pass

class ClusteredFeatureImportanceMDA(FeatureImportanceStrategy):
    def calculate_importance(self, 
                              classifier: RandomForestClassifier,
                              x: pd.DataFrame,
                              y: pd.Series,
                              clusters: Dict[str, List[str]],
                              n_splits: int = 10,
                              score_sample_weights: List[float] = None,
                              train_sample_weights: List[float] = None) -> pd.DataFrame:
        if train_sample_weights is None:
            train_sample_weights = np.ones(x.shape[0])
        if score_sample_weights is None:
            score_sample_weights = np.ones(x.shape[0])

        cv_generator = KFold(n_splits=n_splits)
        score0, score1 = pd.Series(dtype=float), pd.DataFrame(columns=clusters.keys())

        for i, (train, test) in enumerate(cv_generator.split(X=x)):
            print(f"Fold {i} start ...")

            x0, y0, weights0 = x.iloc[train, :], y.iloc[train], train_sample_weights[train]
            x1, y1, weights1 = x.iloc[test, :], y.iloc[test], score_sample_weights[test]

            fit = classifier.fit(X=x0, y=y0, sample_weight=weights0)
            prediction_probability = fit.predict_proba(x1)

            score0[i] = -log_loss(
                y1,
                prediction_probability,
                labels=classifier.classes_,
                sample_weight=weights1
            )

            for j in score1.columns:
                x1_ = x1.copy(deep=True)
                for k in clusters[j]:
                    np.random.shuffle(x1_[k].values)
                prob = fit.predict_proba(x1_)
                score1.loc[i, j] = -log_loss(y1, prob, labels=classifier.classes_)

        importances = (-1 * score1).add(score0, axis=0)
        importances /= (-1 * score1)

        importances = pd.concat({
            "Mean": importances.mean(),
            "StandardDeviation": importances.std() * importances.shape[0] ** -0.5
        }, axis=1)  # Central Limit Theorem

        importances.index = ["C_" + str(i) for i in importances.index]
        return importances
