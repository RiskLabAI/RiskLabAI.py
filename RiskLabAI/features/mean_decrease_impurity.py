import sklearn.datasets as Datasets
import sklearn.ensemble as Ensemble
import pandas as pd
import numpy as np


"""
function: Implementation of an ensemble MDI method
reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
methodology: page 79 Mean-Decrease Impurity section snippet 6.2 (snippet 8.2 2018)
"""
def feature_importance_MDI(
    classifier, # classifier for fit and prediction
    feature_names:list # feature names
) -> pd.DataFrame:
    # feature importance based on IS mean impurity reduction
    dict0 = {i:tree.feature_importances_ for i,tree in enumerate(classifier.estimators_)}

    dataframe0=pd.DataFrame.from_dict(dict0,orient="index")
    dataframe0.columns=feature_names

    dataframe0=dataframe0.replace(0,np.nan) # because max_features=1

    importances=pd.concat({"Mean":dataframe0.mean(), "StandardDeviation":dataframe0.std()*dataframe0.shape[0]**-0.5},axis=1) # CLT
    importances/=importances["Mean"].sum()
    return importances
