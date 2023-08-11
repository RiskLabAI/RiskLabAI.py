import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import yfinance as yf


def inverse_variance_weights(covariance_matrix):
    """
    Compute the inverse-variance portfolio weights.

    Parameters:
    - covariance_matrix (ndarray): Covariance matrix of asset returns.

    Returns:
    - weights (ndarray): Array of portfolio weights.
    """
    inv_var_weights = 1.0 / np.diag(covariance_matrix)
    inv_var_weights /= inv_var_weights.sum()
    return inv_var_weights


def cluster_variance(covariance_matrix, clustered_items):
    """
    Compute the variance of a cluster.

    Parameters:
    - covariance_matrix (ndarray): Covariance matrix of asset returns.
    - clustered_items (list): List of indices of assets in the cluster.

    Returns:
    - cluster_variance (float): Variance of the cluster.
    """
    cov_slice = covariance_matrix.loc[clustered_items, clustered_items]
    weights = inverse_variance_weights(cov_slice).reshape(-1, 1)
    cluster_variance = np.dot(np.dot(weights.T, cov_slice), weights)[0, 0]
    return cluster_variance


def quasi_diagonal(linkage_matrix):
    """
    Return a sorted list of original items to reshape the correlation matrix.

    Parameters:
    - linkage_matrix (ndarray): Linkage matrix obtained from hierarchical clustering.

    Returns:
    - sorted_items (list): Sorted list of original items.
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
        sorted_items = sorted_items.append(dataframe)
        sorted_items = sorted_items.sort_index()
        sorted_items.index = range(sorted_items.shape[0])

    return sorted_items.tolist()


def recursive_bisection(covariance_matrix, sorted_items):
    """
    Compute the Hierarchical Risk Parity (HRP) weights.

    Parameters:
    - covariance_matrix (ndarray): Covariance matrix of asset returns.
    - sorted_items (list): Sorted list of original items.

    Returns:
    - weights (DataFrame): DataFrame of asset weights.
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


def distance_corr(corr_matrix):
    """
    Compute the distance matrix based on correlation.

    Parameters:
    - corr_matrix (ndarray): Correlation matrix.

    Returns:
    - distance_matrix (ndarray): Distance matrix based on correlation.
    """
    distance_matrix = ((1 - corr_matrix) / 2.0) ** 0.5
    return distance_matrix


def plot_corr_matrix(path, corr_matrix, labels=None):
    """
    Plot a heatmap of the correlation matrix.

    Parameters:
    - path (str): Path to save the plot.
    - corr_matrix (ndarray): Correlation matrix.
    - labels (list): List of labels for the assets (optional).
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


def random_data(num_observations, size_uncorr, size_corr, sigma_corr):
    """
    Generate random data.

    Parameters:
    - num_observations (int): Number of observations.
    - size_uncorr (int): Size for uncorrelated data.
    - size_corr (int): Size for correlated data.
    - sigma_corr (float): Standard deviation for correlated data.

    Returns:
    - data (DataFrame): DataFrame of randomly generated data.
    - columns_correlated (list): List of column indices for correlated data.
    """
    np.random.seed(seed=12345)
    random.seed(12345)

    data1 = np.random.normal(0, 1, size=(num_observations, size_uncorr))
    columns_correlated = [random.randint(0, size_uncorr - 1) for _ in range(size_corr)]
    data2 = data1[:, columns_correlated] + np.random.normal(0, sigma_corr, size=(num_observations, len(columns_correlated)))
    data = np.append(data1, data2, axis=1)
    data = pd.DataFrame(data, columns=range(1, data.shape[1] + 1))
    return data, columns_correlated

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import random

def random_data(number_observations, length_sample, size_0, size_1, mu_0, sigma_0, sigma_1):
    """
    Generate random data for Monte Carlo simulation.
    
    Args:
        number_observations (int): Number of observations.
        length_sample (int): Starting point for selecting random observations.
        size_0 (int): Size of uncorrelated data.
        size_1 (int): Size of correlated data.
        mu_0 (float): mu for uncorrelated data.
        sigma_0 (float): sigma for uncorrelated data.
        sigma_1 (float): sigma for correlated data.
    
    Returns:
        tuple: A tuple containing the generated data and the selected columns.
    """
    # Generate random uncorrelated data
    data1 = np.random.normal(mu_0, sigma_0, size=(number_observations, size_0))
    
    # Create correlation between the variables
    columns = [random.randint(0, size_0 - 1) for i in range(size_1)] # randomly select columns
    data2 = data1[:, columns] + np.random.normal(0, sigma_0 * sigma_1, size=(number_observations, len(columns))) # correlated data
    
    # Merge data
    data = np.append(data1, data2, axis=1)
    
    # Add common random shock
    point = np.random.randint(length_sample, number_observations - 1, size=2) # select random observations
    data[np.ix_(point, [columns[0], size_0])] = np.array([[-0.5, -0.5], [2, 2]])
    
    # Add specific random shock
    point = np.random.randint(length_sample, number_observations - 1, size=2) # select random observations
    data[point, columns[-1]] = np.array([-0.5, 2])
    
    return data, columns


def hrp(cov, corr):
    """
    HRP method for constructing a hierarchical portfolio.
    
    Args:
        cov (numpy.ndarray): Covariance matrix.
        corr (numpy.ndarray): Correlation matrix.
    
    Returns:
        pandas.Series: Pandas series containing weights of the hierarchical portfolio.
    """
    # Construct a hierarchical portfolio
    corr_df, cov_df = pd.DataFrame(corr), pd.DataFrame(cov)
    distance = distance_corr(corr_df)
    link = sch.linkage(distance, "single")
    sorted_items = quasi_diagonal(link)
    sorted_items = corr.index[sorted_items].tolist()
    hrp_portfolio = recursive_bisection(cov_df, sorted_items)
    
    return hrp_portfolio.sort_index()


def hrp_mc(number_iters=5000, number_observations=520, size_0=5, size_1=5, mu_0=0, sigma_0=0.01, sigma_1=0.25, length_sample=260, test_size=22):
    """
    Monte Carlo simulation for out of sample comparison of HRP method.
    
    Args:
        number_iters (int): Number of iterations.
        number_observations (int): Number of observations.
        size_0 (int): Size of uncorrelated data.
        size_1 (int): Size of correlated data.
        mu_0 (float): mu for uncorrelated data.
        sigma_0 (float): sigma for uncorrelated data.
        sigma_1 (float): sigma for correlated data.
        length_sample (int): Length for in sample.
        test_size (int): Observation for test set.
    
    Returns:
        None
    """
    methods = [hrp] # methods
    
    results, num_iter = {i.__name__: pd.Series() for i in methods}, 0 # initialize results and number of iterations
    
    pointers = range(length_sample, number_observations, test_size) # pointers for inSample and outSample
    
    while num_iter < number_iters:
        data, columns = random_data(number_observations, length_sample, size_0, size_1, mu_0, sigma_0, sigma_1)
        
        returns = {i.__name__: pd.Series() for i in methods} # initialize returns
        
        for pointer in pointers:
            in_sample = data[pointer - length_sample: pointer] # in sample
            cov_, corr_ = np.cov(in_sample, rowvar=0), np.corrcoef(in_sample, rowvar=0) # cov and corr
            
            out_sample = data[pointer: pointer + test_size] # out of sample
            
            for func in methods:
                weight = func(cov=cov_, corr=corr_) # call methods
                ret = pd.Series(out_sample @ (weight.transpose())) # return
                returns[func.__name__] = returns[func.__name__].append(ret) # update returns
        
        for func in methods:
            ret = returns[func.__name__].reset_index(drop=True) # return column of each method
            cumprod_return = (1 + ret).cumprod() # cumprod of returns
            results[func.__name__].loc[num_iter] = cumprod_return.iloc[-1] - 1 # update results
        
        num_iter += 1 # next iteration
    
    results_df = pd.DataFrame.from_dict(results, orient="columns") # dataframe of results
    results_df.to_csv("results.csv") # csv file
    
    std_results, var_results = results_df.std(), results_df.var() # std and var for each method
    print(pd.concat([std_results, var_results, var_results / var_results["hrp"] - 1], axis=1))


def distance_corr(corr):
    """
    Calculate the distance based on the correlation matrix.
    
    Args:
        corr (pandas.DataFrame): Correlation matrix.
    
    Returns:
        numpy.ndarray: Distance matrix.
    """
    return np.sqrt(2 * (1 - corr))


def quasi_diagonal(link):
    """
    Sort the items based on the linkage matrix.
    
    Args:
        link (numpy.ndarray): Linkage matrix.
    
    Returns:
        list: Sorted items.
    """
    sorted_items = [link[-1, 0], link[-1, 1]]
    
    num_items = link[-1, -2]
    
    while sorted_items[-1] >= num_items:
        index = int(sorted_items[-1] - num_items)
        sorted_items.extend([link[index, 0], link[index, 1]])
    
    return sorted_items


def recursive_bisection(cov, sorted_items):
    """
    Recursive bisection algorithm to calculate portfolio weights.
    
    Args:
        cov (pandas.DataFrame): Covariance matrix.
        sorted_items (list): Sorted items.
    
    Returns:
        pandas.Series: Pandas series containing weights of the hierarchical portfolio.
    """
    if len(sorted_items) > 1:
        cluster = [sorted_items]
        while len(cluster) > 0:
            cluster_ = cluster.pop()
            if len(cluster_) > 1:
                var_covar = cov.loc[cluster_, cluster_]
                e_val, e_vec = np.linalg.eigh(var_covar)
                index = e_val.argsort()[::-1]
                e_val, e_vec = e_val[index], e_vec[:, index]
                w = np.zeros((len(cluster_)))
                w[-1] = 1
                cluster0, cluster1 = [], []
                for i in range(len(cluster_) - 1):
                    d = np.sqrt((w * e_vec[:, i]).T @ e_val[i] @ (w * e_vec[:, i]))
                    u = ((w * e_vec[:, i]) / d).reshape(-1, 1)
                    cluster0.append(cluster_[np.dot(var_covar, u).flatten() <= 0])
                    cluster1.append(cluster_[np.dot(var_covar, u).flatten() > 0])
                cluster.extend([sorted(cluster0), sorted(cluster1)])
            else:
                break
    else:
        cluster = [sorted_items]
    
    weights = pd.Series(1, index=[i for i in cluster[0]])
    
    return weights
