import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import log_loss,accuracy_score

"""
class and functions: splits the data and performes cross validation when observations overlap  
reference: De Prado, M. (2018) Advances in financial machine learning.
methodology: page 109, snippet 7.3
"""
class CrossValidationModel(KFold):
    def __init__(
        self, # The PurgedKFold class containing observations and split information
        n_splits: int, # The number of KFold splits
        times: dict, # Entire observation times
        percent_embargo: float, # Embargo size percentage divided by 100
    ): 
        
        super(CrossValidationModel, self).__init__(n_splits, shuffle=False, random_state=None) # create the PurgedKFold class from Sklearn's KFold
        self.times = times # set the times property in class
        self.percent_embargo = percent_embargo # set the percent_embargo property in class
        
    def purgedkfold_split(self, # The PurgedKFold class containing observations and split information
        data: dict, # The sample that is going be splited
    ):

        test_starts = {}
        for ticker in data.keys():
            test_starts[ticker] = [(i[0], i[-1] + 1) for i in \
                                    np.array_split(np.arange(data[ticker].shape[0]), self.n_splits)] # get all test indices
            
        mono_asset = len(list(data.keys())) == 1    
            
        for split in range(self.n_splits):
            trains = {} 
            tests = {}

            for ticker in data:
                indices = np.arange(data[ticker].shape[0]) # get data positions
                embargo = int(data[ticker].shape[0]*self.percent_embargo) # get embargo size

                start, end = test_starts[ticker][split]

                first_test_index = self.times[ticker].index[start] # get the start of the current test set
                test_indices = indices[start:end] # get test indices for current split
                max_test_index = self.times[ticker].index.searchsorted(self.times[ticker][test_indices].max()) # get the farthest test index
                train_indices = self.times[ticker].index.searchsorted(self.times[ticker][self.times[ticker]<=first_test_index].index) # find the left side of the training data

                if max_test_index + embargo < data[ticker].shape[0]:
                    train_indices = np.concatenate((train_indices, indices[max_test_index + embargo:])) # find the right side of the training data with embargo

                trains[ticker] = np.array(train_indices)
                tests[ticker] = np.array(test_indices)

            if mono_asset:
                trains = trains['ASSET']   
                tests = tests['ASSET']

            yield trains, tests    


    """
    function: purges test observations in the training set
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 106, snippet 7.1
    """        
    @staticmethod
    def purged_train_times(
        data: pd.Series, # Times of entire observations.
        test: pd.Series
    ) -> pd.Series: # Times of testing observations.
        
        # pd.Series.index: Time when the observation started.
        # pd.Series.values: Time when the observation ended.
        
        train_times = data.copy(deep=True) # get a deep copy of train times
        
        for start, end in test.iteritems():
            start_within_test_times = train_times[(start <= train_times.index) & (train_times.index <= end)].index # get times when train starts within test
            end_within_test_times = train_times[(start <= train_times) & (train_times <= end)].index # get times when train ends within test
            envelope_test_times = train_times[(train_times.index <= start) & (end <= train_times)].index # get times when train envelops test
            train_times = train_times.drop(start_within_test_times.union(end_within_test_times).union(envelope_test_times)) # purge observations
            
        return train_times

    """
    function: gets embargo time for each bar
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 108, snippet 7.2
    """

    @staticmethod
    def embargo_times(
        times: pd.Series, # Entire observation times
        percent_embargo: float
    ) -> pd.Series: # Embargo size percentage divided by 100

        step = int(times.shape[0] * percent_embargo) # find the number of embargo bars
        
        if step == 0:
            embargo = pd.Series(times, index=times) # do not perform embargo when the step equals zero
        else:
            embargo = pd.Series(times[step:], index=times[:-step]) # find the embargo time for each time
            embargo = embargo.append(pd.Series(times[-1], index=times[-step:])) # find the embargo time for the last "step" number of bars and join all embargo times
            
        return embargo        
    

    """
    function: uses the PurgedKFold class and functions
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 110, snippet 7.4
    """
    @staticmethod
    def cross_validation_score(
        classifier: ClassifierMixin, # A classifier model
        data: pd.DataFrame, # The sample that is going be splited,
        labels: pd.Series=None, # The labels that are going be splited
        sample_weight: np.ndarray=None, # The sample weights for the classifier
        scoring='neg_log_loss', # Scoring type: ['neg_log_loss','accuracy']
        times: pd.Series=None, # Entire observation times
        n_splits: int=None, # The number of KFold splits,
        cross_validation_generator: BaseCrossValidator=None,
        percent_embargo: float=0.0 # Embargo size percentage divided by 100:
    ) -> np.array:
        
        if scoring not in ['neg_log_loss','accuracy']: # check if the scoring method is correct
            raise Exception('wrong scoring method.') # raise error
        
        if cross_validation_generator is None: # check if the PurgedKFold is nothing
            cross_validation_generator = PurgedKFold(n_splits=n_splits, times=times, percent_embargo=percent_embargo) # initialize
            
        if sample_weight is None: # check if the sample_weight is nothing
            sample_weight = pd.Series(np.ones(len(data))) # initialize    
            
        score = [] # initialize scores
        
        for train, test in cross_validation_generator.split(data):
            fit = classifier.fit(X=data.iloc[train,:], y=labels.iloc[train],
                                sample_weight=sample_weight[train]) # fit model
            
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(data.iloc[test,:]) # predict test
                score_ = -log_loss(labels.iloc[test], prob,
                                sample_weight=sample_weight.iloc[test].values, labels=classifier.classes_) # calculate score
            else:
                pred = fit.predict(data.iloc[test,:]) # predict test
                score_ = accuracy_score(labels.iloc[test], pred, sample_weight=sample_weight.iloc[test].values) # calculate score
                
            score.append(score_) # append score
            
        return np.array(score)    