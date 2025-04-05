# import pandas as pd
# import numpy as np
# from enum import Enum
# from tqdm.notebook import tqdm
# from numpy.random import RandomState
# from collections import Counter
# import itertools
# import sklearn.metrics as Metrics
# from sklearn.preprocessing import minmax_scale

# # -------------------------------
# # Custom ClassificationExperiment
# # -------------------------------
# # This class mimics pycaret’s interface but is implemented entirely with sklearn.
# from sklearn.linear_model import LogisticRegression, RidgeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.dummy import DummyClassifier
# try:
#     from xgboost import XGBClassifier
# except ImportError:
#     XGBClassifier = DummyClassifier
# try:
#     from lightgbm import LGBMClassifier
# except ImportError:
#     LGBMClassifier = DummyClassifier

# class ClassificationExperiment:
#     def setup(self, data, target, fold, train_size, session_id, n_jobs, normalize, normalize_method):
#         # Store the data and parameters; also perform normalization if requested.
#         self.data = data.copy()
#         self.target = target.copy()
#         self.fold = fold
#         self.train_size = train_size
#         self.session_id = session_id
#         self.n_jobs = n_jobs
#         self.normalize = normalize
#         self.normalize_method = normalize_method
#         if normalize:
#             if normalize_method == 'zscore':
#                 self.data = (data - data.mean()) / data.std()
#             elif normalize_method == 'minmax':
#                 self.data = (data - data.min()) / (data.max() - data.min())
#         return self

#     def models(self):
#         # Returns a DataFrame listing available model references.
#         available_models = {
#             'lr': 'lr',
#             'knn': 'knn',
#             'nb': 'nb',
#             'dt': 'dt',
#             'svm': 'svm',
#             'rbfsvm': 'rbfsvm',
#             'gpc': 'gpc',
#             'mlp': 'mlp',
#             'ridge': 'ridge',
#             'rf': 'rf',
#             'qda': 'qda',
#             'ada': 'ada',
#             'gbc': 'gbc',
#             'lda': 'lda',
#             'et': 'et',
#             'xgboost': 'xgboost',
#             'lightgbm': 'lightgbm',
#             'dummy': 'dummy'
#         }
#         df = pd.DataFrame({'Reference': list(available_models.keys())}, 
#                           index=list(available_models.keys()))
#         return df

#     def compare_models(self, n_select, include):
#         # For simplicity, this method selects models from a predefined dict based on the include list.
#         available_models = {
#             'lr': LogisticRegression(),
#             'knn': KNeighborsClassifier(),
#             'nb': GaussianNB(),
#             'dt': DecisionTreeClassifier(),
#             'svm': SVC(probability=True),
#             'rbfsvm': SVC(probability=True, kernel='rbf'),
#             'gpc': GaussianProcessClassifier(),
#             'mlp': MLPClassifier(),
#             'ridge': RidgeClassifier(),
#             'rf': RandomForestClassifier(),
#             'qda': QuadraticDiscriminantAnalysis(),
#             'ada': AdaBoostClassifier(),
#             'gbc': GradientBoostingClassifier(),
#             'lda': LinearDiscriminantAnalysis(),
#             'et': ExtraTreesClassifier(),
#             'xgboost': XGBClassifier(),
#             'lightgbm': LGBMClassifier(),
#             'dummy': DummyClassifier()
#         }
#         selected_models = []
#         for item in include:
#             if isinstance(item, str):
#                 if item in available_models:
#                     selected_models.append(available_models[item])
#             else:
#                 # If already a model instance, include it.
#                 selected_models.append(item)
#         # Select the first n_select models if needed.
#         if isinstance(n_select, int) and n_select < len(selected_models):
#             selected_models = selected_models[:n_select]
#         return selected_models

#     def tune_model(self, model, return_tuner=False, fold=4, verbose=True, tuner_verbose=True, 
#                    optimize='Accuracy', search_library='scikit-optimize', search_algorithm='bayesian', 
#                    n_iter=25, choose_better=True, custom_grid=None):
#         # A minimal tuning implementation.
#         from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#         if custom_grid is not None and len(custom_grid) > 0:
#             tuner = GridSearchCV(model, custom_grid, cv=fold)
#             tuner.fit(self.data, self.target)
#             tuned_model = tuner.best_estimator_
#         else:
#             tuner = None
#             tuned_model = model
#         if return_tuner:
#             return tuned_model, tuner
#         else:
#             return tuned_model

#     def finalize_model(self, estimator, model_only=True):
#         # Dummy finalize: simply returns the estimator.
#         return estimator

# # -------------------------------
# # Feature Importance Functions
# # -------------------------------
# # (For models that require the feature names, we add a check to set classifier.feature_names_in_ if needed.)
# def feature_importance_linear_models(
#     classifier: 'sklearn.base.BaseEstimator',
#     X: pd.DataFrame, 
#     y,
# ) -> pd.Series:
#     classifier.fit(X, y)
#     if isinstance(X, pd.DataFrame) and not hasattr(classifier, 'feature_names_in_'):
#         classifier.feature_names_in_ = X.columns
#     coefficients_importances = np.abs(classifier.coef_).mean(axis=0)
#     coefficients_importances = pd.Series(coefficients_importances, classifier.feature_names_in_)
#     importances_scaled = minmax_scale(
#         coefficients_importances,
#         feature_range=(0, 1),
#         axis=0,
#     )
#     return pd.Series(importances_scaled, index=coefficients_importances.index)

# def feature_importance_sklearn(
#     classifier: 'sklearn.base.BaseEstimator',
#     X: pd.DataFrame, 
#     y,
# ) -> pd.Series:
#     classifier.fit(X, y)
#     if isinstance(X, pd.DataFrame) and not hasattr(classifier, 'feature_names_in_'):
#         classifier.feature_names_in_ = X.columns
#     importances = pd.Series(classifier.feature_importances_, index=classifier.feature_names_in_)
#     importances = pd.concat({
#         "Mean": importances.mean(),
#     }, axis=1)
#     importances_scaled = minmax_scale(
#         importances["Mean"],
#         feature_range=(0, 1),
#         axis=0,
#     )
#     return pd.Series(importances_scaled, index=importances.index)

# def feature_importance_RFE(
#     classifier: 'sklearn.base.BaseEstimator',
#     X: pd.DataFrame, 
#     y,
# ) -> pd.Series:
#     from sklearn.feature_selection import RFE
#     rfe = RFE(
#         estimator=classifier,
#         verbose=0,
#         n_features_to_select=1,    
#     )
#     rfe.fit(X, y)
#     if isinstance(X, pd.DataFrame) and not hasattr(rfe, 'feature_names_in_'):
#         rfe.feature_names_in_ = X.columns
#     inverted_ranking = np.max(rfe.ranking_) - rfe.ranking_ + 1
#     normalized_importance = minmax_scale(inverted_ranking)
#     return pd.Series(normalized_importance, index=rfe.feature_names_in_)

# def feature_importance_SFI(
#     classifier: 'sklearn.base.BaseEstimator',
#     X: pd.DataFrame, 
#     y,
#     n_splits: int = 5,
#     score_sample_weights: np.ndarray = None,  
#     train_sample_weights: np.ndarray = None, 
# ) -> pd.Series:
#     from sklearn.model_selection import StratifiedKFold 
#     if train_sample_weights is None:
#         train_sample_weights = np.ones(X.shape[0])
#     if score_sample_weights is None:
#         score_sample_weights = np.ones(X.shape[0])
#     cv_generator = StratifiedKFold(n_splits=n_splits)
#     feature_names = X.columns
#     importances = []
#     for feature_name in feature_names:
#         scores = []
#         for train, test in cv_generator.split(X, y):
#             x_train, y_train, weights_train = X.iloc[train, :][[feature_name]], y.iloc[train], train_sample_weights[train]
#             x_test, y_test, weights_test = X.iloc[test, :][[feature_name]], y.iloc[test], score_sample_weights[test]
#             try:
#                 classifier.fit(x_train, y_train, sample_weight=weights_train)
#             except:
#                 classifier.fit(x_train, y_train)
#             prediction = classifier.predict(x_test)
#             score = Metrics.accuracy_score(y_test, prediction, sample_weight=weights_test)
#             scores.append(score)
#         importances.append({
#             "FeatureName": feature_name,
#             "Mean": np.mean(scores),
#         })
#     importances = pd.DataFrame(importances)
#     importances_scaled = minmax_scale(
#         importances["Mean"],
#         feature_range=(0, 1),
#         axis=0,
#     )
#     return pd.Series(importances_scaled, index=importances.FeatureName)

# def feature_importance_MDI(
#     classifier: 'sklearn.base.BaseEstimator',
#     X: pd.DataFrame, 
#     y,
# ) -> pd.Series:
#     classifier.fit(X, y)
#     if isinstance(X, pd.DataFrame) and not hasattr(classifier, 'feature_names_in_'):
#         classifier.feature_names_in_ = X.columns
#     importances = pd.Series(classifier.feature_importances_, index=classifier.feature_names_in_)
#     importances_scaled = minmax_scale(
#         importances,
#         feature_range=(0, 1),
#         axis=0,
#     )
#     return pd.Series(importances_scaled, index=importances.index)

# def feature_importance_MDA(
#     classifier: 'sklearn.base.BaseEstimator',
#     X: pd.DataFrame, 
#     y,
#     n_repeats: int = 5,
# ) -> pd.Series:
#     from sklearn.inspection import permutation_importance
#     classifier.fit(X, y)
#     importances = permutation_importance(classifier, X, y, n_repeats=n_repeats, random_state=43).importances_mean
#     importances = minmax_scale(
#         importances,
#         feature_range=(0, 1),
#         axis=0,
#     )
#     return pd.Series(importances, index=X.columns)

# # -------------------------------
# # Univariate Analysis Functions
# # -------------------------------
# class UAMeasure(Enum):
#     SPEARMAN = "Spearman"
#     PEARSON = "Pearson"
#     KENDAL_TAU = "Kendal-Tau"
#     MUTUAL_INFORMATION = "Mutual Information"
#     ANOVA_F = "ANOVA F-Stat"

# def feature_importance_univariate_analysis(measurements, X, y):
#     from scipy.stats import pearsonr, kendalltau
#     from sklearn.feature_selection import mutual_info_classif, f_classif
#     ua = pd.DataFrame(data=None, index=X.columns)
#     if UAMeasure.SPEARMAN.value in measurements:
#         spearmans = minmax_scale(X.apply(lambda feature: np.abs(np.corrcoef(feature, y)[0, 1]), axis=0))
#         ua[UAMeasure.SPEARMAN.value] = spearmans
#     if UAMeasure.PEARSON.value in measurements:
#         pearsons = minmax_scale(X.apply(lambda feature: np.abs(pearsonr(feature, y).statistic), axis=0))
#         ua[UAMeasure.PEARSON.value] = pearsons
#     if UAMeasure.KENDAL_TAU.value in measurements:
#         kendalltaus = minmax_scale(X.apply(lambda feature: np.abs(kendalltau(feature, y).statistic), axis=0))
#         ua[UAMeasure.KENDAL_TAU.value] = kendalltaus
#     if UAMeasure.MUTUAL_INFORMATION.value in measurements:
#         mutual_informations = minmax_scale(mutual_info_classif(X, y, n_neighbors=51))
#         ua[UAMeasure.MUTUAL_INFORMATION.value] = mutual_informations
#     if UAMeasure.ANOVA_F.value in measurements:
#         f_values = minmax_scale(f_classif(X, y)[0])
#         ua[UAMeasure.ANOVA_F.value] = f_values
#     return ua

# def full_name(klass):
#     return ".".join([klass.__module__, klass.__name__])

# # -------------------------------
# # Enums for FI Model Types
# # -------------------------------
# class FIModel(Enum):
#     COEFFICIENT_BASED = "Coefficient-Based + Shrinkage & Selection"
#     TREE_BASED = "Tree-Based"
#     PERMUTATION_BASED = "Permutation-Based"
#     SINGLE_FEATURE_BASED = "Single Feature-Based"
#     RFE_BASED = "Recursive Feature Elimination-Based"

# # -------------------------------
# # SWEFI Class
# # -------------------------------
# class SWEFI:
#     def __init__(self, X: pd.DataFrame, y, n_fold:int =10):
#         self.percentage = None
#         # Use our custom ClassificationExperiment instead of pycaret
#         self.clfx = ClassificationExperiment().setup(data=X, target=y, fold=n_fold, train_size=0.99, 
#                                                        session_id=123, n_jobs=-1, normalize=True, normalize_method='zscore')
#         self.X = X
#         self.y = y

#     def select_models(self, select_n_model: 'Union[int, float]' = 100):
#         model_to_methods = self.clfx.models()
#         model_to_methods['FI Methods'] = pd.Series({
#             'lr': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value, FIModel.SINGLE_FEATURE_BASED.value, FIModel.RFE_BASED.value],
#             'knn': [FIModel.PERMUTATION_BASED.value],
#             'nb': [FIModel.SINGLE_FEATURE_BASED.value],
#             'dt': [FIModel.SINGLE_FEATURE_BASED.value],
#             'svm': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value],
#             'rbfsvm': [FIModel.PERMUTATION_BASED.value],
#             'gpc': [FIModel.PERMUTATION_BASED.value],
#             'mlp': [FIModel.PERMUTATION_BASED.value],
#             'ridge': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value, FIModel.SINGLE_FEATURE_BASED.value, FIModel.RFE_BASED.value],
#             'rf': [FIModel.TREE_BASED.value, FIModel.RFE_BASED.value],
#             'qda': [FIModel.PERMUTATION_BASED.value],
#             'ada': [FIModel.PERMUTATION_BASED.value],
#             'gbc': [FIModel.PERMUTATION_BASED.value],
#             'lda': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value, FIModel.SINGLE_FEATURE_BASED.value, FIModel.RFE_BASED.value],
#             'et': [FIModel.TREE_BASED.value, FIModel.RFE_BASED.value],
#             'xgboost': [FIModel.TREE_BASED.value, FIModel.RFE_BASED.value],
#             'lightgbm': [FIModel.PERMUTATION_BASED.value],
#             'dummy': [FIModel.PERMUTATION_BASED.value],
#         })
#         model_to_methods = model_to_methods[['Reference', 'FI Methods']].set_index('Reference').squeeze().to_dict()
#         model_to_methods.update({
#             "sklearn.svm._classes.SVC": [FIModel.COEFFICIENT_BASED.value, FIModel.RFE_BASED.value],
#         })
#         self.model_to_methods = model_to_methods
#         exclude = set(['dummy', 'svm', 'knn', 'nb', 'lightgbm', 'ada', 'gpc', 'rbfsvm'])
#         all_models = set(self.clfx.models().index.tolist())
#         external_models = [SVC(probability=True, kernel='linear')]
#         include = list(all_models - exclude) + external_models
#         if type(select_n_model) == float:
#             self.learning_models = self.clfx.compare_models(n_select=round(len(self.clfx.models()) * select_n_model),
#                                                             include=include)
#         else:
#             self.learning_models = self.clfx.compare_models(n_select=select_n_model,
#                                                             include=include)
#         return self

#     def select_univariate_analysis_measurements(self, measurements: list = ['Pearson', 'Spearman', 'Kendal']):
#         self.measurements = measurements
#         return self

#     def fine_tune_selected_models(self, hpo_n_fold:int=4, hpo_n_iter:int=25, hpo_metric:str='Accuracy', 
#                                   hpo_search_library:str='scikit-optimize', hpo_search_algorithm='bayesian'):
#         learning_models = self.learning_models
#         model_to_methods = self.model_to_methods
#         model_to_custom_config = {
#             "sklearn.ensemble._forest.ExtraTreesClassifier": None,
#             "sklearn.neighbors._classification.KNeighborsClassifier": None,
#             "sklearn.ensemble._forest.RandomForestClassifier": None,
#             "xgboost.sklearn.XGBClassifier": None,
#             "lightgbm.sklearn.LGBMClassifier": {},
#             "sklearn.neural_network._multilayer_perceptron.MLPClassifier": None,
#             "sklearn.ensemble._gb.GradientBoostingClassifier": {},
#             "sklearn.naive_bayes.GaussianNB": {},
#             "sklearn.ensemble._weight_boosting.AdaBoostClassifier": {},
#             "sklearn.linear_model._ridge.RidgeClassifier": {
#                 "alpha": [1],
#                 "solver": ["lsqr"],
#             },
#             "sklearn.discriminant_analysis.LinearDiscriminantAnalysis": {
#                 "shrinkage": [0.0],
#                 "solver": ["lsqr"],
#             },
#             "sklearn.linear_model._logistic.LogisticRegression": {
#                 "C": [1],
#                 "solver": ['liblinear'],
#             },
#             "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis": {},
#             "sklearn.tree._classes.DecisionTreeClassifier": None,
#             "sklearn.svm._classes.SVC": {
#                 'probability': [True],
#                 'C': [1.0],
#                 'kernel': ["linear"],
#             },
#         }
#         tuned_learning_models = []
#         for model in learning_models:
#             print("-" * 80)
#             print(model.__class__.__name__)
#             if model_to_custom_config[full_name(model.__class__)] is None:
#                 tlm, tuner = self.clfx.tune_model(
#                     model,
#                     return_tuner=True,
#                     fold=hpo_n_fold, 
#                     verbose=True, 
#                     tuner_verbose=True, 
#                     optimize=hpo_metric, 
#                     search_library=hpo_search_library, 
#                     search_algorithm=hpo_search_algorithm,
#                     n_iter=hpo_n_iter,
#                     choose_better=True
#                 )
#             elif len(model_to_custom_config[full_name(model.__class__)]) == 0:
#                 print("No hyper-tuning ...")
#                 tlm = model
#             else:
#                 print("Custom config ...")
#                 tlm, tuner = self.clfx.tune_model(
#                     model,
#                     return_tuner=True,
#                     fold=hpo_n_fold, 
#                     verbose=True, 
#                     tuner_verbose=True, 
#                     optimize=hpo_metric, 
#                     search_library='scikit-learn', 
#                     search_algorithm='grid',
#                     custom_grid=model_to_custom_config[full_name(model.__class__)], 
#                     n_iter=hpo_n_iter,
#                     choose_better=True
#                 )
#             tuned_learning_models.append(tlm)
#         self.finalized_tuned_learning_models = [self.clfx.finalize_model(estimator=tlm, model_only=True) for tlm in tuned_learning_models]
#         model_method_pairs = []
#         for tlm in tuned_learning_models:
#             methods = model_to_methods[full_name(tlm.__class__)]
#             for method in methods:
#                 model_method_pairs.append((tlm, method))
#         self.model_method_pairs = model_method_pairs
#         return self

#     def _initialize_model_feature_importance_data(self, models: list, n_iteration: int = 5):
#         feature_names = self.X.columns
#         index_columns = ["Model Name", "Feature Importance Model", "Subset Index"]
#         to_apply_product_list = [
#             models,
#             list(range(n_iteration)),
#         ]
#         tuples = [(type(model).__name__, proper_fi_model, iteration) 
#                   for (model, proper_fi_model), iteration in itertools.product(*to_apply_product_list)]
#         index = pd.MultiIndex.from_tuples(tuples, names=index_columns)
#         data = pd.DataFrame({}, columns=feature_names, index=index)
#         return data, index

#     def _initialize_univariate_analysis_feature_importance_data(self, measurements: list, n_iteration: int = 5):
#         feature_names = self.X.columns
#         index_columns = ["Model Name", "Feature Importance Model", "Subset Index"]
#         to_apply_product_list = [
#             measurements,
#             list(range(n_iteration)),
#         ]
#         tuples = [("Univariate Analysis", ua_measurement, iteration) 
#                   for (ua_measurement,), iteration in itertools.product(*to_apply_product_list)]
#         index = pd.MultiIndex.from_tuples(tuples, names=index_columns)
#         data = pd.DataFrame({}, columns=feature_names, index=index)
#         return data, index

#     @staticmethod
#     def stationary_bootstrap(X: pd.DataFrame, y, n_iteration: int = 5):
#         from arch.bootstrap import optimal_block_length, StationaryBootstrap
#         optimal_block_size = round(optimal_block_length(X).mean(axis=0)["stationary"])
#         random_state = RandomState(1234)
#         bootstraper = StationaryBootstrap(optimal_block_size, X=X, y=y, random_state=random_state)
#         for bootstraped_data in bootstraper.bootstrap(n_iteration):
#             yield bootstraped_data[1]['X'], bootstraped_data[1]['y']

#     @staticmethod
#     def iid_bootstrap(X: pd.DataFrame, y, n_iteration: int = 5):
#         from arch.bootstrap import IIDBootstrap
#         random_state = RandomState(1234)
#         bootstraper = IIDBootstrap(X=X, y=y, random_state=random_state)
#         for bootstraped_data in bootstraper.bootstrap(n_iteration):
#             yield bootstraped_data[1]['X'], bootstraped_data[1]['y']

#     def compute_feature_importance_data(self, bootstrap_method, n_iteration: int = 10, n_repeats: int = 10):
#         X = self.X
#         y = self.y
#         X = (X - X.mean()) / X.std()
#         model_method_pairs = self.model_method_pairs
#         models_fi, models_index = self._initialize_model_feature_importance_data(model_method_pairs, n_iteration=n_iteration)
#         measurements = self.measurements
#         measurements_fi, measurements_index = self._initialize_univariate_analysis_feature_importance_data(measurements, n_iteration=n_iteration)
#         for iteration, (X_train, y_train) in tqdm(enumerate(bootstrap_method(X, y, n_iteration)), total=n_iteration):
#             for model, proper_fi_model in model_method_pairs:
#                 if proper_fi_model == FIModel.TREE_BASED.value:
#                     try:
#                         result = feature_importance_MDI(model, X_train, y_train).values
#                     except:
#                         result = feature_importance_sklearn(model, X_train, y_train).values
#                 elif proper_fi_model == FIModel.PERMUTATION_BASED.value:
#                     result = feature_importance_MDA(model, X_train, y_train, n_repeats=n_repeats).values
#                 elif proper_fi_model == FIModel.COEFFICIENT_BASED.value:
#                     result = feature_importance_linear_models(model, X_train, y_train).values
#                 elif proper_fi_model == FIModel.SINGLE_FEATURE_BASED.value:
#                     result = feature_importance_SFI(model, X_train, y_train, n_splits=n_repeats).values
#                 elif proper_fi_model == FIModel.RFE_BASED.value:
#                     result = feature_importance_RFE(model, X_train, y_train).values
#                 result = (result - result.min()) / (result.max() - result.min()) 
#                 models_fi.loc[type(model).__name__, proper_fi_model, iteration] = result
#             ua_result = feature_importance_univariate_analysis(measurements, X_train, y_train).T.values
#             measurements_fi.loc[pd.IndexSlice["Univariate Analysis", :, iteration]] = ua_result
#         self.models_fi = models_fi
#         self.models_index = models_index
#         self.measurements_fi = measurements_fi
#         self.measurements_index = measurements_index
#         self.n_iterationn = n_iteration
#         return self

#     def compute_swefi_scores(self, percentage: float = 0.5, weight: str = 'linear'):
#         features = self.X.columns
#         fi = pd.concat([self.models_fi, self.measurements_fi])
#         index = self.models_index.tolist() + self.measurements_index.tolist()
#         k = round(len(features) * percentage)
#         important_features_together = []
#         for idx in index:
#             important_features_in_current_step = fi.loc[idx].squeeze().sort_values(ascending=False)[:k].index.tolist()
#             important_features_together.extend(important_features_in_current_step)
#         feature_ranked_as_k_top_important = dict(zip(features, [0] * len(features)))
#         n_times_that_feature_ranked_as_important = Counter(important_features_together)
#         dict.update(feature_ranked_as_k_top_important, n_times_that_feature_ranked_as_important)
#         feature_ranked_as_k_top_important = pd.Series(feature_ranked_as_k_top_important)
#         stability_scores = feature_ranked_as_k_top_important / (self.n_iterationn * (len(self.model_method_pairs) + len(self.measurements)))
#         if weight == 'linear':
#             weights = stability_scores / stability_scores.sum()
#             swefi = (fi * weights)
#         elif weight == 'logarithmic':
#             weights = np.log(stability_scores + 1)
#             weights = weights / weights.sum()
#             swefi = (fi * weights)
#         elif weight == 'exponential':
#             weights = np.exp(stability_scores)
#             weights = weights / weights.sum()
#             swefi = (fi * weights)
#         elif weight == 'harmonic':
#             swefi = (2 * fi * stability_scores) / (fi + stability_scores)
#         elif weight == 'power-2':
#             weights = stability_scores ** 2
#             weights = weights / weights.sum()
#             swefi = (fi * weights)
#         elif weight == 'power-0.5':
#             weights = (stability_scores ** 0.5)
#             weights = weights / weights.sum()
#             swefi = (fi * weights)
#         elif weight == 'entropy':
#             weights = (-stability_scores * np.log(stability_scores))
#             weights = weights / weights.sum()            
#             swefi = (fi * weights)
#         else:
#             raise NotImplementedError(f"Weight {weight} not implemented.")
#         swefi = pd.concat(
#             [swefi.mean() / swefi.mean().sum(), swefi.std() * (swefi.shape[0]) ** -0.5 / swefi.mean().sum()],
#             axis=1,
#         ).rename(columns={0: 'mean(SWEFI)', 1: 'std(SWEFI)'}).sort_values(by='mean(SWEFI)')
#         self.stability_scores = stability_scores
#         self.swefi = swefi
#         return self

#     def get_swefi_scores(self):
#         return self.swefi

#     def get_inner_models_feature_importances(self):
#         models_fi = self.models_fi.groupby(level=[0, 1]).sum()
#         models_fi = models_fi.div(models_fi.sum(axis=1).values, axis=0).transpose()
#         return models_fi
