from typing import Union, List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from itertools import combinations

class CombinatorialPurgedKFold(KFold):
    def __init__(
            self,
            n_total_splits: int,
            n_test_splits: int,
            samples_info_sets: Union[pd.Series, dict] = None,
            percentage_embargo: float = 0.0
    ) -> None:
        """
        Initialize a CombinatorialPurgedKFold object.

        :param n_total_splits: Total number of splits.
        :type n_total_splits: int

        :param n_test_splits: Number of testing splits.
        :type n_test_splits: int

        :param samples_info_sets: Information sets for samples.
        :type samples_info_sets: Union[pd.Series, dict]

        :param percentage_embargo: Percentage of embargo.
        :type percentage_embargo: float

        :return: None
        """
        if not isinstance(samples_info_sets, (pd.Series, dict)):
            raise ValueError('The samples_info_sets parameter must be a pd.Series or dictionary')

        if not isinstance(samples_info_sets, dict):
            samples_info_sets = {'ASSET': samples_info_sets}

        super().__init__(n_total_splits)

        self.samples_info_sets = samples_info_sets
        self.percentage_embargo = percentage_embargo
        self.n_test_splits = n_test_splits   
    
    def _purge_embargo_splits(
            self,
            data: Union[pd.DataFrame, dict]
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Purge and apply embargo to the splits.

        :param data: Input data for splits.
        :type data: Union[pd.DataFrame, dict]

        :return: Tuple of all training and test splits and backtest paths.
        :rtype: Tuple[Dict[str, List[int]], Dict[str, List[int]]]
        """
        if not isinstance(data, dict):
            data = {'ASSET': data}

        first_asset = list(data.keys())[0]
        for asset, asset_data in data.items():
            if asset_data.shape[0] != self.samples_info_sets[asset].shape[0]:
                raise ValueError("data and the samples_info_sets series must be the same length")

        test_ranges = {}
        backtest_paths = {}
        splits_indices = {}
        combinatorial_test_ranges = {}
        num_backtest_paths = backtest_paths_number(self.n_splits, self.n_test_splits)

        for asset in data:
            test_ranges[asset] = [(ix[0], ix[-1] + 1) for ix in
                                         np.array_split(np.arange(data[asset].shape[0]), self.n_splits)]
            backtest_paths[asset] = {} 
            splits_indices[asset] = {}

            for index, [start_ix, end_ix] in enumerate(test_ranges[asset]):
                splits_indices[asset][index] = [start_ix, end_ix]

            # Possible test splits for each fold
            combinatorial_splits = list(combinations(list(splits_indices[asset].keys()), self.n_test_splits))
            combinatorial_test_ranges[asset] = []  # List of test indices formed from combinatorial splits

            for combination in combinatorial_splits:
                temp_test_indices = []  # Array of test indices for current split combination
                for int_index in combination:
                    temp_test_indices.append(splits_indices[asset][int_index])

                combinatorial_test_ranges[asset].append(temp_test_indices)    

            # Prepare backtest paths
            backtest_paths[asset] = {}
            for i in range(num_backtest_paths):
                path = []
                for split_idx in splits_indices[asset].values():
                    path.append({'Train': None, 'Test': split_idx})

                backtest_paths[asset][f'Path {i + 1}'] = path    


        embargo = int(data.shape[0] * self.percentage_embargo)

        all_trains = []
        all_tests = []

        for i in range(len(combinatorial_test_ranges[first_asset])):
            trains = {} 
            tests = {}

            for asset in data:
                embargo: int = int(data[asset].shape[0] * self.percentage_embargo)
                test_splits = combinatorial_test_ranges[asset][i]
                # Embargo
                test_times = pd.Series(index=[self.samples_info_sets[asset][ix[0]] for ix in test_splits], data=[
                    max(self.samples_info_sets[asset][ix[0]: ix[1]]) if ix[1] - 1 + embargo >= data[asset].shape[0] else
                    max(self.samples_info_sets[asset][ix[0]: ix[1] + embargo]) for ix in test_splits])

                # Purge
                train_times = purged_train_times(self.samples_info_sets[asset], test_times)

                # Get indices
                train_indices = []
                for train_ix in train_times.index:
                    train_indices.append(self.samples_info_sets[asset].index.get_loc(train_ix))    

                trains[asset] = train_indices
                tests[asset] = test_splits

            all_trains.append(trains)
            all_tests.append(tests)     

        return zip(all_trains, all_tests), backtest_paths         

    def backtest_paths_splits(
            self,
            data: Union[pd.DataFrame, dict]
    ) -> dict:
        """
        Generate backtest paths splits.

        :param data: Input data for splits.
        :type data: Union[pd.DataFrame, dict]

        :return: Dictionary of backtest paths.
        :rtype: dict
        """
        is_dict = isinstance(data, dict)
        splits, backtest_paths = self._purge_embargo_splits(data)

        for trains, tests in splits:
            for asset in data:
                # Fill backtest paths using train/test splits from CPCV
                for split in tests[asset]:
                    found = False  # Flag indicating that split was found and filled in one of backtest paths
                    for path in backtest_paths[asset]:
                        for path_el in path.values():
                            if path_el['Train'] is None and split[0] == path_el['Test'][0] and found is False:
                                path_el['Train'] = np.array(trains[asset])
                                path_el['Test'] = list(range(split[0], split[-1]))
                                found = True  

        if not is_dict:
            backtest_paths = backtest_paths['ASSET']                        

        return backtest_paths

    def split(
            self,
            data: Union[pd.DataFrame, dict],
            labels: Union[pd.Series, dict] = None,
            groups=None
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Split data into training and test sets.

        :param data: Input data for splits.
        :type data: Union[pd.DataFrame, dict]

        :param labels: Target labels.
        :type labels: Union[pd.Series, dict]

        :param groups: Grouping of data. 
        :type groups: Any

        :return: Tuple of training and test indices.
        :rtype: Tuple[Dict[str, List[int]], Dict[str, List[int]]]
        """
        is_dict = isinstance(data, dict)
        splits, _ = self._purge_embargo_splits(data)

        for trains, tests in splits:
            test_indices = {}

            for asset in data:
                test_indices[asset] = []
                for [start_ix, end_ix] in tests[asset]:
                    test_indices[asset].extend(list(range(start_ix, end_ix)))

            if not is_dict:
                trains = trains['ASSET']
                test_indices = test_indices['ASSET']

            yield trains, test_indices  

    def backtest_paths_predictions(
            self,
            data: Union[pd.DataFrame, dict],
            labels: Union[pd.Series, dict],
            model: Union[BaseEstimator, dict],
            sample_weight: Union[np.ndarray, dict] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate predictions using backtest paths.

        :param data: Input data for prediction.
        :type data: Union[pd.DataFrame, dict]

        :param labels: Target labels.
        :type labels: Union[pd.Series, dict]

        :param model: Model used for prediction.
        :type model: Union[BaseEstimator, dict]

        :param sample_weight: Weights of samples.
        :type sample_weight: Union[np.ndarray, dict]

        :return: Predictions for each path.
        :rtype: Dict[str, pd.Series]
        """
        is_dict = isinstance(data, dict)
        backtest_paths = self.backtest_paths_splits(data)
        paths = {}

        if not isinstance(labels, dict):
            labels = {'ASSET' : labels}

        if not isinstance(model, dict):
            model = {'ASSET' : model}

        if not isinstance(sample_weight, dict):
            sample_weight = {'ASSET' : sample_weight}  

        if not isinstance(backtest_paths, dict):
            backtest_paths = {'ASSET' : backtest_paths}          

        for asset in data: 
            paths[asset] = pd.DataFrame()

            for path_key in backtest_paths[asset].keys():
                path = backtest_paths[asset][path_key]
                predictions = np.array([])

                for split in path:
                    X_train, y_train, weights_train = data[asset].iloc[split['Train']], labels[asset].iloc[split['Train']] \
                                                    , sample_weight[asset][split['Train']]
                    X_test = data[asset].iloc[split['Test']]

                    model[asset].fit(X=X_train, y=y_train, sample_weight=weights_train)
                    y_pred = model[asset].predict(X_test)
                    predictions = np.append(predictions, y_pred)

                paths[asset][path_key] = pd.Series(predictions, index=labels[asset].index)

        if not is_dict:
            paths = paths['ASSET']

        return paths