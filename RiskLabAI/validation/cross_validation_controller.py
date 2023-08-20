# TODO: from Model_7 import * what's that?
# import pandas as pd
# import numpy as np
# from Model_7 import *

# """
# class and functions: splits the data and performes cross validation when observations overlap  
# reference: De Prado, M. (2018) Advances in financial machine learning.
# methodology: page 109, snippet 7.3
# """

# class PurgedKFold(CrossValidationModel):
#     def __init__(self, # The PurgedKFold class containing observations and split information
#                  n_splits: int=3, # The number of KFold splits
#                  times: Union[pd.Series, dict]=None, # Entire observation times
#                  percent_embargo: float=0.0): # Embargo size percentage divided by 100
        
#         if type(times) is not Union[pd.Series, dict]: # check if times parameter is a pd.Series or dict
#             raise ValueError('The \"times\" argument must be a pd.Series or dictionary') # raise error 

#         if type(times) is not dict:
#             times = {'ASSET' : times}   
        
#         super(CrossValidationModel, self).__init__(n_splits, times, percent_embargo) # create the PurgedKFoldModel class from Model
        
#     def split(self, # The PurgedKFold class containing observations and split information
#               data: Union[pd.DataFrame, dict], # The sample that is going be splited
#               labels: pd.Series=None, # The labels that are going be splited
#               groups=None): # Group our labels
        
#         if not isinstance(data, dict):
#             data = {'ASSET' : data}                      
                    
#         for ticker in data:
#             if data[ticker].shape[0] != self.times[ticker].shape[0]:
#                 raise ValueError("The \"data\" and the \"times\" arguments must be the same length")
        
#         super().purgedkfold_split(data)


# class PurgedEmbargoMethods(CrossValidationModel):
#     """
#     function: purges test observations in the training set
#     reference: De Prado, M. (2018) Advances in financial machine learning.
#     methodology: page 106, snippet 7.1
#     """        

#     @staticmethod
#     def purged_train_times(
#         data: pd.Series, # Times of entire observations.
#         test: pd.Series
#     ) -> pd.Series: # Times of testing observations.
        
#         # pd.Series.index: Time when the observation started.
#         # pd.Series.values: Time when the observation ended.

#         super().purged_train_times(data, test)
        

#     """
#     function: gets embargo time for each bar
#     reference: De Prado, M. (2018) Advances in financial machine learning.
#     methodology: page 108, snippet 7.2
#     """

#     @staticmethod
#     def embargo_times(
#         times: pd.Series, # Entire observation times
#         percent_embargo: float
#     ) -> pd.Series: # Embargo size percentage divided by 100

#         super().embargo_times(times, percent_embargo) 

#     """
#     function: uses the PurgedKFold class and functions
#     reference: De Prado, M. (2018) Advances in financial machine learning.
#     methodology: page 110, snippet 7.4
#     """
#     @staticmethod
#     def cross_validation_score(
#         classifier: ClassifierMixin, # A classifier model
#         data: Union[pd.DataFrame, dict], # The sample that is going be splited,
#         labels: Union[pd.Series, dict]=None, # The labels that are going be splited
#         sample_weight: Union[np.ndarray, dict]=None, # The sample weights for the classifier
#         scoring='neg_log_loss', # Scoring type: ['neg_log_loss','accuracy']
#         times: pd.Series=None, # Entire observation times
#         n_splits: int=None, # The number of KFold splits,
#         cross_validation_generator: BaseCrossValidator=None,
#         percent_embargo: float=0.0 # Embargo size percentage divided by 100:
#     ) -> np.array:

#         super().cross_validation_score(classifier, data, labels, sample_weight, scoring, \
#                                        times, n_splits, cross_validation_generator, percent_embargo)     
