
import multiprocessing as mp
import pandas as pd
import numpy as np
import datetime as dt
import time

"""
function: reporting progress of computing 
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 20.9
"""
def reportProgress(jobNumber,nJobs,startTime,task):
    # Report progress as asynch jobs are completed
    massage=[float(jobNumber)/nJobs,(time.time()-startTime)/60.]   #compute remaining time 
    massage.append(massage[1]*(1/massage[0]-1)) #append remaining time
    timeStamp=str(dt.datetime.fromtimestamp(time.time())) #local time 
    massage=timeStamp+' '+str(round(massage[0]*100,2))+'% '+task+' done after '+ str(round(massage[1],2))+' minutes. Remaining '+str(round(massage[2],2))+' minutes.' #create massage
    if jobNumber<nJobs:
        print(massage+'\r')
    else:
        print(massage+'\n')
    return
"""
function: process jobs 
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 20.9
"""
def processJobs(jobs,task=None,nThreads=24):
    # Run in parallel.
    # jobs must contain a ’func’ callback, for expandCall
    if task is None:
        task=jobs[0]['func'].__name__ #initial func
    pool=mp.Pool(processes=nThreads) #thread pool
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time() #initial task to processors 
   
    # Process asynchronous output, report progress
   
    for i,out_ in enumerate(outputs):
        out.append(out_) 
        reportProgress(i,len(jobs),time0,task)
    pool.close()
    pool.join() # this is needed to prevent memory leaks
    return out

"""
function: passing the job (molecule) to the callback function
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 20.9
"""
def expandCall(kargs):
# Expand the arguments of a callback function, kargs[’func’]
    func=kargs['func'] 
    del kargs['func']
    output=func(**kargs)
    return output

 
"""
function: single-thread execution, for debugging
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 20.8
"""
def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    output=[]
    for job in jobs:
        output_=expandCall(job)
        output.append(output_)
    return output

"""
function: the linearPartitions function
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 20.5
"""
def linearPartitions(nAtoms,nThreads):
    # partition of atoms with a single loop
    partitions=np.linspace(0,nAtoms,min(nThreads,nAtoms)+1) # split [0...nAtoms) to partition 
    partitions=np.ceil(partitions).astype(int) #intigerize(!) number
    return partitions
    
"""
function: the nestedPartitions function
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 20.6
"""
def nestedPartitions(nAtoms,nThreads,upperTriangle=False):
    # partition of atoms with an inner loop
    partitions,nThreads_=[0],min(nThreads,nAtoms)
    for num in range(nThreads_):
        #compute parts size according book in page formula in  page 308
        partitions=1 + 4*(partitions[-1]**2+partitions[-1]+nAtoms*(nAtoms+1.)/nThreads_)
        partitions=(-1+partitions**.5)/2.
        partitions.append(partitions)
    partitions=np.round(partitions).astype(int)
    if upperTriangle: # the first rows are the heaviest
        partitions=np.cumsum(np.diff(partitions)[::-1])
        partitions=np.append(np.array([0]),partitions)
    return partitions

"""
function: the mpPandasObj
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 20.7

Parallelize jobs, return a DataFrame or Series
+ func: function to be parallelized. Returns a DataFrame
+ pdObj[0]: Name of argument used to pass the molecule
+ pdObj[1]: List of atoms that will be grouped into molecules
+ kargs: any other argument needed by func
Example: df1=mpPandasObj(func,(’molecule’,df0.index),24,**kargs)
"""
def mpPandasObj(function,pandasObject,nThreads=2,mpBatches=1,linearPartition = True,**kargs):
    if linearPartition: #check which partition algorithm most used 
        parts=linearPartitions(len(pandasObject[1]),nThreads*mpBatches)
    else:
        parts=nestedPartitions(len(pandasObject[1]),nThreads*mpBatches)

    jobs=[] 
    for i in range(1,len(parts)):
        job={pandasObject[0]:pandasObject[1][parts[i-1]:parts[i]],'func':function} #create job 
        job.update(kargs) #append arguments of func to job 
        jobs.append(job) # append job to jobs 
    if nThreads==1: #check number of threads 
        out=processJobs_(jobs)
    else:
        out=processJobs(jobs,numThreads=nThreads)

    if isinstance(out[0],pd.DataFrame): # check type of out
        df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):
        df0=pd.Series()
    else:
        return out
    # according to output of func sort index because  multiprocessing approach reorder output!
    for i in out:
        df0=df0.append(i)
    df0=df0.sort_index()
    return df0