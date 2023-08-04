from Python_Chapter_7 import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    def fit(self,X,y,sample_weight=None,**fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight']=sample_weight
        return super(MyPipeline,self).fit(X,y,**fit_params)


"""
    function: hyper parameter tuning 
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 131, snippet 9.3
"""
def clfHyperFit(
        feature_data, # data of features 
        label, # labels of data 
        time, # observation time
        pipe_clf, # our estimator
        param_grid, # parameter space 
        cv=3, # number of group for cross validation 
        bagging=[0,-1,1.], # bagging type
        rndSearchIter=0,
        n_jobs=-1,
        percent_embargo=0, # percent of embergo
        **fit_params
    ):
    
    if set(label.values)=={0,1}:
        scoring='f1' # f1 for meta-labeling
    else:
        scoring='neg_log_loss' # symmetric towards all cases
    
    #1) hyperparameter search, on train data
    inner_cv=PurgedKFold(n_splits=cv,times=time,percent_embargo=percent_embargo) # purged
    if rndSearchIter==0:
        gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs)
    else:
        gs=RandomizedSearchCV(estimator=pipe_clf,param_distributions= param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,n_iter=rndSearchIter)
    gs=gs.fit(feature_data,label,**fit_params).best_estimator_ # pipeline
    
    #2) fit validated model on the entirety of the data
    if bagging[1]>0:
        gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),max_samples=float(bagging[1]),max_features=float(bagging[2]),n_jobs=n_jobs)
        gs=gs.fit(feature_data,label,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs=Pipeline([('bag',gs)])

    return gs