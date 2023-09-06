import pandas as pd
from cross_validator_factory import CrossValidatorFactory
from remove_overlapping_train_times import remove_overlapping_train_times

class CrossValidatorController:
    """
    Controller class to handle the cross-validation process.
    """

    def __init__(
            self, 
            validator_type: str,
            n_splits: int,
            times: pd.Series,
            percent_embargo: float = 0.0
    ):
        """
        Initialize the CrossValidatorController.

        :param str validator_type: Type of the cross-validator ('purged' or 'combinatorial_purged').
        :param int n_splits: Number of KFold splits.
        :param pd.Series times: Series representing the entire observation times.
        :param float percent_embargo: Embargo size as a percentage divided by 100.
        """
        self.cross_validator = CrossValidatorFactory.create_cross_validator(
            validator_type,
            n_splits,
            times,
            percent_embargo
        )

    def run_cross_validation(
            self, 
            data: pd.DataFrame, 
            labels: pd.Series
    ):
        """
        Execute the cross-validation process.

        :param pd.DataFrame data: Data to be used in cross-validation.
        :param pd.Series labels: Labels to be used in cross-validation.
        """
        for train_indices, test_indices in self.cross_validator.split(data, labels):
            train_times = self.cross_validator.times.iloc[train_indices]
            test_times = self.cross_validator.times.iloc[test_indices]
            
            # Remove overlapping times in training data based on testing data intervals
            filtered_train_times = remove_overlapping_train_times(train_times, test_times)
            
            # Extract the filtered training and test datasets
            filtered_train_data = data.iloc[filtered_train_times.index]
            test_data = data.iloc[test_indices]
            
            # Your machine learning logic goes here
            
            # For example,
            # model = train_model(filtered_train_data, labels.iloc[filtered_train_times.index])
            # evaluate_model(model, test_data, labels.iloc[test_indices])

# Usage
# controller = CrossValidatorController('purged', 5, times_series, 0.1)
# controller.run_cross_validation(data_dataframe, labels_series)
