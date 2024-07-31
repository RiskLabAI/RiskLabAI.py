import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.datasets as Datasets


def generate_ar_time_series(n, ar_params, ma_params, sigma=1.0):
    """
    Generate an autoregressive (AR) time series.

    Parameters:
    - n: Length of the time series
    - ar_params: List of AR parameters [phi1, phi2, ..., phi_p]
    - sigma: Standard deviation of the white noise

    Returns:
    - time_series: Generated AR time series
    """
    # Create the AR coefficients for the model
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]

    # Generate the ARMA process
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    time_series = arma_process.generate_sample(nsample=n, scale=sigma)

    return time_series


def create_lagged_data(time_series, k):
    """
    Create a DataFrame with the time series and its k lags.

    Parameters:
    - time_series: The generated time series
    - k: Number of lags

    Returns:
    - df: DataFrame containing the time series and its lags
    """
    data = {'y': time_series}
    for i in range(1, k + 1):
        data[f'lag_{i}'] = np.roll(time_series, i)

    df = pd.DataFrame(data)
    df = df.iloc[k:]  # Drop the initial rows with NaN values due to the roll

    return df.to_numpy()

def generate_cross_sectional_dataset(
    n_informative: int = 5, 
    n_redundant: int = 25,  
    n_noise: int = 5, 
    n_samples: int = 10000,  
    random_state: int = 41,  
    sigma_std: float = 0.1, 
    n_clusters_per_class: int = 2,
) -> tuple:
    n_features = n_informative + n_redundant + n_noise
    np.random.seed(random_state)
    X, y = Datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features-n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state,
        n_clusters_per_class=n_clusters_per_class,
    )

    columns = [f"I{i}" for i in range(n_informative)]
    columns += [f"N{i}" for i in range(n_noise)]
    X, y = pd.DataFrame(X, columns=columns), pd.Series(y)
    i = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X[f"R{k} (from I{j})"] = X[f"I{j}"] + np.random.normal(size=X.shape[0]) * sigma_std

    return X, y

def generate_multivariate_time_series_dataset(
    n_informative: int = 10,  # number of informative features
    n_redundant: int = 10,  # number of redundant features
    n_noise: int = 20,
    n_samples: int = 10000,  # number of sample to generate
    n_time_steps: int = 2, # Two business years
    random_state: int = 41,  # random state
    sigma_std: float = 0.1,  # standard deviation of generation
    time_series_params = [[0.3], [0.3], 0.1], # AR parameter (phi1)
):
    n_lag = n_time_steps - 1

    informative_features = np.zeros(shape=(n_samples, n_informative, n_time_steps))
    for k in range(n_informative):
        time_series = generate_ar_time_series(n_samples + n_lag, time_series_params[0], time_series_params[1], 1)
        informative_features[:, k, :] = create_lagged_data(time_series, n_lag)

    np.random.seed(random_state)
    i = np.random.choice(range(n_informative), size=n_redundant)
    np.random.seed()
    
    linear_redundant_features = np.zeros((n_samples, n_redundant, n_time_steps))
    linear_redundant_features_from_ = []

    for k, j in enumerate(i):
        linear_redundant_features[:, k, :] = informative_features[:, j, :] + np.random.normal(size=(n_samples, n_time_steps)) * sigma_std
        linear_redundant_features_from_.append(j)

    # Generate noise features
    noise_features = np.zeros(shape=(n_samples, n_noise, n_time_steps))
    for k in range(n_noise):
        time_series = generate_ar_time_series(n_samples + n_lag, 0.5, 0.5, 1)
        noise_features[:, k, :] = create_lagged_data(time_series, n_lag)

    # Concatenate all features
    X = np.concatenate([
        informative_features,
        linear_redundant_features,
        noise_features],
        axis=1)

    columns = [f"I{i} lag{j}" for i in range(n_informative) for j in range(n_time_steps)]
    columns += [f"R{i} from I{k} lag{j}" for i, k in enumerate(linear_redundant_features_from_) for j in range(n_time_steps)]
    columns += [f"N{i} lag{j}" for i in range(n_noise) for j in range(n_time_steps)]

    X = X.reshape((n_samples, -1))
    X = pd.DataFrame(X, columns=columns)

    I = X.iloc[:, :n_informative * n_time_steps]
    I = (I - I.mean()) / I.std()
    weights = np.ones(shape=len(I.columns))

    # Calculate linear combination of features plus some noise
    linear_combination = I @ weights
    # Generate binary target variable based on probabilities
    y = (linear_combination > 0).astype(int)

    y = pd.Series(y)

    return X, y

