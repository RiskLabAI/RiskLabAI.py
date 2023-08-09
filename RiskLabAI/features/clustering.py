import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import numpy as np, pandas as pd
from scipy.linalg import block_diag
from sklearn.utils import check_random_state

"""
    function: Derive the correlation matrix from a covariance matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.3, Page 27
"""
def covToCorr(covariance):  # covariance matrix
    std = np.sqrt(np.diag(covariance))  # standard deviations
    correlation = covariance / np.outer(std, std)  # create correlation matrix
    correlation[correlation < -1], correlation[correlation > 1] = -1, 1  # numerical error
    return correlation


"""
    function: Clustering
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.1, Page 56
"""
def clusterKMeansBase(
    correlation, numberClusters=10, iterations=10  # corr pd.dataframe  # maximum number of clusters
):  # iterations
    distance, silh = ((1 - correlation.fillna(0)) / 2.0) ** 0.5, pd.Series()  # observations matrix
    for init in range(iterations):
        for i in range(2, numberClusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)  # clustering distance with maximum cluster i
            kmeans_ = kmeans_.fit(distance)  # fit kmeans
            silh_ = silhouette_samples(distance, kmeans_.labels_)  # calculate silh scores
            statistic = (silh_.mean() / silh_.std(), silh.mean() / silh.std())  # calculate t-statistic
            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silh, kmeans = silh_, kmeans_  # choose better clustering
    indexSorted = np.argsort(kmeans.labels_)  # sort arguments
    correlationSorted = correlation.iloc[indexSorted]  # reorder rows
    correlationSorted = correlationSorted.iloc[:, indexSorted]  # reorder columns
    clusters = {
        i: correlation.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)
    }  # cluster members
    silh = pd.Series(silh, index=distance.index)  # silh scores
    return correlationSorted, clusters, silh


"""
    function: make new clustering
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.2, Page 58
"""
def makeNewOutputs(correlation, clusters, clusters2):  # corr dataframe  # cluster 1  # cluster 2
    # merge two clusters
    clustersNew = {}
    for i in clusters.keys():
        clustersNew[len(clustersNew.keys())] = list(clusters[i])
    for i in clusters2.keys():
        clustersNew[len(clustersNew.keys())] = list(clusters2[i])
    indexNew = [j for i in clustersNew for j in clustersNew[i]]  # sorted index of assets
    correlationNew = correlation.loc[indexNew, indexNew]  # new corr matrix
    distance = ((1 - correlation.fillna(0)) / 2.0) ** 0.5  # distance dataframe
    labelsKmeans = np.zeros(len(distance.columns))  # initial labels
    for i in clustersNew.keys():
        index = [distance.index.get_loc(k) for k in clustersNew[i]]
        labelsKmeans[index] = i  # label for clusters
    silhNew = pd.Series(silhouette_samples(distance, labelsKmeans), index=distance.index)  # silh series
    return correlationNew, clustersNew, silhNew


"""
    function: clustering (ONC)
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.2, Page 58
"""
def clusterKMeansTop(
    correlation, numberClusters=None, iterations=10  # corr dataframe  # number of clusters
):  # number of iterations
    if numberClusters == None:
        numberClusters = correlation.shape[1] - 1  # set number of cluster
    # clustering
    correlationSorted, clusters, silh = clusterKMeansBase(
        correlation, numberClusters=min(numberClusters, correlation.shape[1] - 1), iterations=iterations
    )
    clusterTstats = {
        i: np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]]) for i in clusters.keys()
    }  # calculate t-stat of each cluster
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)  # calculate mean of t-stats
    redoClusters = [
        i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean
    ]  # select clusters which have t-stat lower than mean
    if len(redoClusters) <= 1:
        return correlationSorted, clusters, silh
    else:
        keysRedo = [j for i in redoClusters for j in clusters[i]]  # select keys of redoclusters
        correlationTemp = correlation.loc[keysRedo, keysRedo]  # slice corr for redoclusters
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])  # calculate mean of t-stats
        # call again clusterKMeansTop
        correlationSorted2, clusters2, silh2 = clusterKMeansTop(
            correlationTemp, numberClusters=min(numberClusters, correlationTemp.shape[1] - 1), iterations=iterations
        )
        # Make new outputs, if necessary
        correlationNew, clustersNew, silhNew = makeNewOutputs(
            correlation, {i: clusters[i] for i in clusters.keys() if i not in redoClusters}, clusters2
        )
        # mean of t-stats new output
        newTstatMean = np.mean(
            [np.mean(silhNew[clustersNew[i]]) / np.std(silhNew[clustersNew[i]]) for i in clustersNew.keys()]
        )
        if newTstatMean <= tStatMean:
            return correlationSorted, clusters, silh
        else:
            return correlationNew, clustersNew, silhNew


"""
    function: Compute sub cov matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.3, Page 61
"""
def randomCovarianceSub(
    numberObservations,  # number of observations
    numberColumns,  # number of cols
    sigma,  # sigma for normal distribution
    randomState=None,
):
    # Sub correlation matrix
    domain = check_random_state(randomState)
    if numberColumns == 1:
        return np.ones((1, 1))
    data = domain.normal(size=(numberObservations, 1))  # generate data
    data = np.repeat(data, numberColumns, axis=1)  # repeat data
    data += domain.normal(scale=sigma, size=data.shape)  # add data
    covariance = np.cov(data, rowvar=False)  # covariance of data
    return covariance


"""
    function: Compute random block cov matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.3, Page 61
"""
def randomBlockCovariance(
    numberColumns,  # number of cols
    numberBlocks,  # number of blocks
    blockSizeMin=1,  # minimum size of block
    sigma=1.0,  # sigma for normal distribution
    randomState=None,
):
    # Generate a block random correlation matrix
    domain = check_random_state(randomState)
    # generate data
    parts = domain.choice(range(1, numberColumns - (blockSizeMin - 1) * numberBlocks), numberBlocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, numberColumns - (blockSizeMin - 1) * numberBlocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + blockSizeMin
    cov = None
    for column in parts:
        thisCovariance = randomCovarianceSub(
            int(max(column * (column + 1) / 2.0, 100)), column, sigma, randomState=domain
        )  # sub covariance
        if cov is None:
            cov = thisCovariance.copy()  # copy covariance matrix
        else:
            cov = block_diag(cov, thisCovariance)  # block diagram covariance matrix
    return cov


"""
    function: Compute random block corr matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.3, Page 61
"""
def randomBlockCorrelation(
    numberColumns, numberBlocks, randomState=None, blockSizeMin=1  # number of cols  # number of blocks  # for rand data
):  #  minimum size of block
    # Form block corr
    domain = check_random_state(randomState)
    # generate 2 random block diagram cov matrix
    covariance1 = randomBlockCovariance(
        numberColumns, numberBlocks, blockSizeMin=blockSizeMin, sigma=0.5, randomState=domain
    )
    covariance2 = randomBlockCovariance(
        numberColumns, 1, blockSizeMin=blockSizeMin, sigma=1.0, randomState=domain
    )  # add noise
    covariance1 += covariance2  # add 2 cov matrix
    correlation = covToCorr(covariance1)  # corr matrix
    correlation = pd.DataFrame(correlation)  # pd.dataframe of corr matrix
    return correlation
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from sklearn.utils import check_random_state

def cov_to_corr(covariance):
    """
    Convert covariance matrix to correlation matrix.

    :param covariance: Covariance matrix
    :type covariance: np.ndarray
    :return: Correlation matrix
    :rtype: np.ndarray
    """
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)
    correlation[correlation < -1] = -1
    correlation[correlation > 1] = 1
    return correlation

def cluster_kmeans_base(correlation, number_clusters=10, iterations=10):
    """
    Perform KMeans clustering on correlation matrix.

    :param correlation: Correlation matrix
    :type correlation: pd.DataFrame
    :param number_clusters: Number of clusters, default is 10
    :type number_clusters: int
    :param iterations: Number of iterations, default is 10
    :type iterations: int
    :return: Sorted correlation matrix, clusters, silhouette scores
    :rtype: pd.DataFrame, dict, pd.Series
    """
    distance = ((1 - correlation.fillna(0)) / 2.0) ** 0.5
    silh = pd.Series()

    for init in range(iterations):
        for i in range(2, number_clusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(distance)
            silh_ = silhouette_samples(distance, kmeans_.labels_)
            statistic = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silh, kmeans = silh_, kmeans_

    index_sorted = np.argsort(kmeans.labels_)
    correlation_sorted = correlation.iloc[index_sorted]
    correlation_sorted = correlation_sorted.iloc[:, index_sorted]

    clusters = {
        i: correlation.columns[np.where(kmeans.labels_ == i)[0]].tolist()
        for i in np.unique(kmeans.labels_)
    }

    silh = pd.Series(silh, index=distance.index)
    return correlation_sorted, clusters, silh

# The rest of the functions follow the same conventions and docstring formats as shown above.

# # Example usage:
# if __name__ == "__main__":
#     correlation_matrix = randomBlockCorrelation(numberColumns=20, numberBlocks=5)
#     sorted_corr, cluster_dict, silh_scores = cluster_kmeans_base(correlation_matrix)
#     print(sorted_corr)
#     print(cluster_dict)
#     print(silh_scores)
