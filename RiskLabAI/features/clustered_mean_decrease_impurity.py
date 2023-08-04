import pandas as pd
import numpy as np

"""
    function: Group Mean and Standard Deviation
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 86 Clustered MDI section
"""
def group_mean_std(
    dataframe0: pd.DataFrame,  # input dataframe
    clusters: dict  # clusters
) -> pd.DataFrame:

    output = pd.DataFrame(columns=["Mean", "StandardDeviation"])

    for (cluster_index, j) in clusters.items():
        dataframe1 = dataframe0[j].sum(axis=1)
        output.loc["C_"+str(cluster_index), "Mean"] = dataframe1.mean()
        output.loc["C_"+str(cluster_index), "StandardDeviation"] = dataframe1.std()*dataframe1.shape[0]**-0.5

    return output


"""
    function: Clustered feature importance MDI
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 86 Clustered MDI section
"""
def clustered_feature_importance_MDI(
    classifier,  # classifier for mdi
    feature_names,  # feature names
    clusters  # clusters
) -> pd.DataFrame:

    dict0 = {i: tree.feature_importances_ for i, tree in enumerate(classifier.estimators_)}
    dataframe0 = pd.DataFrame.from_dict(dict0, orient="index")
    dataframe0.columns = feature_names
    dataframe0 = dataframe0.replace(0, np.nan)  # because max_features=1

    importances = group_mean_std(dataframe0, clusters)
    importances /= importances["Mean"].sum()
    return importances
