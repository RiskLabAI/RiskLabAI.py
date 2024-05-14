import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import yfinance as yf
import random


def inverse_variance_weights(covariance_matrix: pd.DataFrame) -> np.ndarray:
    """
    Compute the inverse-variance portfolio weights.

    :param covariance_matrix: Covariance matrix of asset returns.
    :type covariance_matrix: pd.DataFrame
    :return: Array of portfolio weights.
    :rtype: np.ndarray
    """
    inv_var_weights = 1.0 / np.diag(covariance_matrix)
    inv_var_weights /= inv_var_weights.sum()
    return inv_var_weights


def cluster_variance(covariance_matrix: pd.DataFrame, clustered_items: list) -> float:
    """
    Compute the variance of a cluster.

    :param covariance_matrix: Covariance matrix of asset returns.
    :type covariance_matrix: pd.DataFrame
    :param clustered_items: List of indices of assets in the cluster.
    :type clustered_items: list
    :return: Variance of the cluster.
    :rtype: float
    """
    cov_slice = covariance_matrix.loc[clustered_items, clustered_items]
    weights = inverse_variance_weights(cov_slice).reshape(-1, 1)
    cluster_variance = np.dot(np.dot(weights.T, cov_slice), weights)[0, 0]
    return cluster_variance


def quasi_diagonal(linkage_matrix: np.ndarray) -> list:
    """
    Return a sorted list of original items to reshape the correlation matrix.

    :param linkage_matrix: Linkage matrix obtained from hierarchical clustering.
    :type linkage_matrix: np.ndarray
    :return: Sorted list of original items.
    :rtype: list
    """
    linkage_matrix = linkage_matrix.astype(int)
    sorted_items = pd.Series([linkage_matrix[-1, 0], linkage_matrix[-1, 1]])
    num_items = linkage_matrix[-1, 3]

    while sorted_items.max() >= num_items:
        sorted_items.index = range(0, sorted_items.shape[0] * 2, 2)
        dataframe = sorted_items[sorted_items >= num_items]
        i = dataframe.index
        j = dataframe.values - num_items
        sorted_items[i] = linkage_matrix[j, 0]
        dataframe = pd.Series(linkage_matrix[j, 1], index=i + 1)
        sorted_items = sorted_items._append(dataframe)
        sorted_items = sorted_items.sort_index()
        sorted_items.index = range(sorted_items.shape[0])

    return sorted_items.tolist()


def recursive_bisection(covariance_matrix: pd.DataFrame, sorted_items: list) -> pd.Series:
    """
    Compute the Hierarchical Risk Parity (HRP) weights.

    :param covariance_matrix: Covariance matrix of asset returns.
    :type covariance_matrix: pd.DataFrame
    :param sorted_items: Sorted list of original items.
    :type sorted_items: list
    :return: DataFrame of asset weights.
    :rtype: pd.Series
    """
    weights = pd.Series(1, index=sorted_items)
    clustered_items = [sorted_items]

    while len(clustered_items) > 0:
        clustered_items = [
            i[j:k] for i in clustered_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1
        ]

        for i in range(0, len(clustered_items), 2):
            clustered_items_0 = clustered_items[i]
            clustered_items_1 = clustered_items[i + 1]
            cluster_variance_0 = cluster_variance(covariance_matrix, clustered_items_0)
            cluster_variance_1 = cluster_variance(covariance_matrix, clustered_items_1)
            alpha = 1 - cluster_variance_0 / (cluster_variance_0 + cluster_variance_1)
            weights[clustered_items_0] *= alpha
            weights[clustered_items_1] *= 1 - alpha

    return weights

def distance_corr(
        corr_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute the distance matrix based on correlation.

    :param corr_matrix: Correlation matrix.
    :type corr_matrix: np.ndarray
    :return: Distance matrix based on correlation.
    :rtype: np.ndarray
    """
    distance_matrix = ((1 - corr_matrix) / 2.0) ** 0.5
    return distance_matrix


def plot_corr_matrix(
        path: str,
        corr_matrix: np.ndarray,
        labels: list = None
) -> None:
    """
    Plot a heatmap of the correlation matrix.

    :param path: Path to save the plot.
    :type path: str
    :param corr_matrix: Correlation matrix.
    :type corr_matrix: np.ndarray
    :param labels: List of labels for the assets (optional).
    :type labels: list, optional
    """
    if labels is None:
        labels = []

    plt.figure()
    plt.pcolor(corr_matrix)
    plt.colorbar()
    plt.yticks(np.arange(0.5, corr_matrix.shape[0] + 0.5), labels)
    plt.xticks(np.arange(0.5, corr_matrix.shape[0] + 0.5), labels)
    plt.savefig(path)
    plt.clf()
    plt.close()


def random_data(
        num_observations: int,
        size_uncorr: int,
        size_corr: int,
        sigma_corr: float
) -> (pd.DataFrame, list):
    """
    Generate random data.

    :param num_observations: Number of observations.
    :type num_observations: int
    :param size_uncorr: Size for uncorrelated data.
    :type size_uncorr: int
    :param size_corr: Size for correlated data.
    :type size_corr: int
    :param sigma_corr: Standard deviation for correlated data.
    :type sigma_corr: float
    :return: DataFrame of randomly generated data and list of column indices for correlated data.
    :rtype: pd.DataFrame, list
    """
    np.random.seed(seed=12345)
    random.seed(12345)

    data1 = np.random.normal(0, 1, size=(num_observations, size_uncorr))
    columns_correlated = [random.randint(0, size_uncorr - 1) for _ in range(size_corr)]
    data2 = data1[:, columns_correlated] + np.random.normal(0, sigma_corr, size=(num_observations, len(columns_correlated)))
    data = np.append(data1, data2, axis=1)
    data = pd.DataFrame(data, columns=range(1, data.shape[1] + 1))
    return data, columns_correlated

def random_data2(
        number_observations: int,
        length_sample: int,
        size_uncorrelated: int,
        size_correlated: int,
        mu_uncorrelated: float,
        sigma_uncorrelated: float,
        sigma_correlated: float
) -> (np.ndarray, list):
    """
    Generate random data for Monte Carlo simulation.

    :param number_observations: Number of observations.
    :type number_observations: int
    :param length_sample: Starting point for selecting random observations.
    :type length_sample: int
    :param size_uncorrelated: Size of uncorrelated data.
    :type size_uncorrelated: int
    :param size_correlated: Size of correlated data.
    :type size_correlated: int
    :param mu_uncorrelated: mu for uncorrelated data.
    :type mu_uncorrelated: float
    :param sigma_uncorrelated: sigma for uncorrelated data.
    :type sigma_uncorrelated: float
    :param sigma_correlated: sigma for correlated data.
    :type sigma_correlated: float
    :return: A tuple containing the generated data and the selected columns.
    :rtype: np.ndarray, list
    """
    # Generate random uncorrelated data
    data1 = np.random.normal(mu_uncorrelated, sigma_uncorrelated, size=(number_observations, size_uncorrelated))

    # Create correlation between the variables
    columns = [random.randint(0, size_uncorrelated - 1) for i in range(size_correlated)]  # randomly select columns
    data2 = data1[:, columns] + np.random.normal(0, sigma_uncorrelated * sigma_correlated, size=(number_observations, len(columns)))  # correlated data

    # Merge data
    data = np.append(data1, data2, axis=1)

    # Add common random shock
    points = np.random.randint(length_sample, number_observations - 1, size=2)  # select random observations
    data[np.ix_(points, [columns[0], size_uncorrelated])] = np.array([[-0.5, -0.5], [2, 2]])

    # Add specific random shock
    points = np.random.randint(length_sample, number_observations - 1, size=2)  # select random observations
    data[points, columns[-1]] = np.array([-0.5, 2])

    return data, columns


def hrp(cov: np.ndarray, corr: np.ndarray) -> pd.Series:
    """
    HRP method for constructing a hierarchical portfolio.
    
    :param cov: Covariance matrix.
    :type cov: np.ndarray
    :param corr: Correlation matrix.
    :type corr: np.ndarray
    :return: Pandas series containing weights of the hierarchical portfolio.
    :rtype: pd.Series
    """
    # Construct a hierarchical portfolio
    corr_df, cov_df = pd.DataFrame(corr), pd.DataFrame(cov)
    distance = distance_corr(corr_df)
    link = sch.linkage(distance, "single")
    sorted_items = quasi_diagonal(link)
    sorted_items = corr_df.index[sorted_items].tolist()
    hrp_portfolio = recursive_bisection(cov_df, sorted_items)
    
    return hrp_portfolio.sort_index()


def hrp_mc(
        number_iterations: int = 5000,
        number_observations: int = 520,
        size_uncorrelated: int = 5,
        size_correlated: int = 5,
        mu_uncorrelated: float = 0,
        sigma_uncorrelated: float = 0.01,
        sigma_correlated: float = 0.25,
        length_sample: int = 260,
        test_size: int = 22
) -> None:
    """
    Monte Carlo simulation for out of sample comparison of HRP method.
    
    :param number_iterations: Number of iterations.
    :type number_iterations: int
    :param number_observations: Number of observations.
    :type number_observations: int
    :param size_uncorrelated: Size of uncorrelated data.
    :type size_uncorrelated: int
    :param size_correlated: Size of correlated data.
    :type size_correlated: int
    :param mu_uncorrelated: mu for uncorrelated data.
    :type mu_uncorrelated: float
    :param sigma_uncorrelated: sigma for uncorrelated data.
    :type sigma_uncorrelated: float
    :param sigma_correlated: sigma for correlated data.
    :type sigma_correlated: float
    :param length_sample: Length for in sample.
    :type length_sample: int
    :param test_size: Observation for test set.
    :type test_size: int
    :return: None
    """
    methods = [hrp]  # methods
    
    results, iteration_counter = {i.__name__: pd.Series() for i in methods}, 0  # initialize results and iteration counter
    
    pointers = range(length_sample, number_observations, test_size)  # pointers for in-sample and out-sample
    
    while iteration_counter < number_iterations:
        data, columns = random_data2(number_observations, length_sample, size_uncorrelated, size_correlated, mu_uncorrelated, sigma_uncorrelated, sigma_correlated)
        
        returns = {i.__name__: pd.Series() for i in methods}  # initialize returns
        
        for pointer in pointers:
            in_sample = data[pointer - length_sample: pointer]  # in sample
            cov_, corr_ = np.cov(in_sample, rowvar=0), np.corrcoef(in_sample, rowvar=0)  # cov and corr
            
            out_sample = data[pointer: pointer + test_size]  # out of sample
            
            for func in methods:
                weight = func(cov=cov_, corr=corr_)  # call methods
                ret = pd.Series(out_sample @ (weight.transpose()))  # return
                returns[func.__name__] = returns[func.__name__]._append(ret)  # update returns
        
        for func in methods:
            ret = returns[func.__name__].reset_index(drop=True)  # return column of each method
            cumprod_return = (1 + ret).cumprod()  # cumprod of returns
            results[func.__name__].loc[iteration_counter] = cumprod_return.iloc[-1] - 1  # update results
        
        iteration_counter += 1  # next iteration
    
    results_df = pd.DataFrame.from_dict(results, orient="columns")  # dataframe of results
    results_df.to_csv("results.csv")  # csv file
    
    std_results, var_results = results_df.std(), results_df.var()  # std and var for each method
    print(pd.concat([std_results, var_results, var_results / var_results["hrp"] - 1], axis=1))


# def distance_corr(corr: pd.DataFrame) -> np.ndarray:
#     """
#     Calculate the distance based on the correlation matrix.
#
#     :param corr: Correlation matrix.
#     :type corr: pandas.DataFrame
#     :return: Distance matrix.
#     :rtype: numpy.ndarray
#
#     The distance is computed based on the formula:
#
#     .. math:: \text{distance} = \sqrt{2 \cdot (1 - \text{corr})}
#     """
#     return np.sqrt(2 * (1 - corr))
#
#
# def quasi_diagonal(link: np.ndarray) -> list:
#     """
#     Sort the items based on the linkage matrix.
#
#     :param link: Linkage matrix.
#     :type link: numpy.ndarray
#     :return: Sorted items.
#     :rtype: list
#     """
#     sorted_items = [link[-1, 0], link[-1, 1]]
#
#     num_items = link[-1, -2]
#
#     while sorted_items[-1] >= num_items:
#         index = int(sorted_items[-1] - num_items)
#         sorted_items.extend([link[index, 0], link[index, 1]])
#
#     return sorted_items
#
#
# def recursive_bisection(cov: pd.DataFrame, sorted_items: list) -> pd.Series:
#     """
#     Recursive bisection algorithm to calculate portfolio weights.
#
#     :param cov: Covariance matrix.
#     :type cov: pandas.DataFrame
#     :param sorted_items: Sorted items.
#     :type sorted_items: list
#     :return: Pandas series containing weights of the hierarchical portfolio.
#     :rtype: pandas.Series
#     """
#     if len(sorted_items) > 1:
#         cluster = [sorted_items]
#         while len(cluster) > 0:
#             cluster_ = cluster.pop()
#             if len(cluster_) > 1:
#                 var_covar = cov.loc[cluster_, cluster_]
#                 e_val, e_vec = np.linalg.eigh(var_covar)
#                 index = e_val.argsort()[::-1]
#                 e_val, e_vec = e_val[index], e_vec[:, index]
#                 w = np.zeros((len(cluster_)))
#                 w[-1] = 1
#                 cluster0, cluster1 = [], []
#                 for i in range(len(cluster_) - 1):
#                     d = np.sqrt((w * e_vec[:, i]).T @ e_val[i] @ (w * e_vec[:, i]))
#                     u = ((w * e_vec[:, i]) / d).reshape(-1, 1)
#                     cluster0.append(cluster_[np.dot(var_covar, u).flatten() <= 0])
#                     cluster1.append(cluster_[np.dot(var_covar, u).flatten() > 0])
#                 cluster.extend([sorted(cluster0), sorted(cluster1)])
#             else:
#                 break
#     else:
#         cluster = [sorted_items]
#
#     weights = pd.Series(1, index=[i for i in cluster[0]])
#
#     return weights
