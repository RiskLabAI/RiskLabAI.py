import pandas as pd
import numpy as np 
import datetime
import time
import sys
from scipy import stats
from statsmodels import*
from sklearn.neighbors._kde import KernelDensity
import scipy
from scipy.optimize import *
from sklearn.covariance import LedoitWolf
from scipy.linalg import block_diag

def marcenkoPasturPDF(var, # variance of observations
                      q, # T/N
                      pts) : #
    # Marcenko-Pastur pdf
    emin, emax = var*(1 - (1./q)**.5)**2, var*(1 + (1./q)**.5)**2
    eval = np.linspace(emin, emax, pts).flatten()
    pdf = q/(2*np.pi*var*eval)*((emax - eval)*(eval - emin))**.5
    pdf = pd.Series(pdf, index = eval)
    return pdf

def PCA(matrix): # Get eval,evec from a Hermitian matrix
    import numpy as np, pandas as pd
    eval, evec = np.linalg.eigh(matrix)
    indices = eval.argsort()[::-1] # arguments for sorting eval desc
    eval, evec = eval[indices], evec[:,indices]
    eval = np.diagflat(eval)
    return eval, evec

def fitKDE(obs, # Series of observations
           bwidth = .25, #
           kernel = 'gaussian', # 
           x = None): #
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) == 1:obs=obs.reshape(-1,1)
    kde = KernelDensity(kernel = kernel, bandwidth = bwidth).fit(obs)
    if x is None: x = np.unique(obs).reshape(-1,1)
    if len(x.shape) == 1:x = x.reshape(-1,1)
    logprob = kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logprob), index=x.flatten())
    return pdf

def randomCov(ncols, #
              nfacts): #
    w = np.random.normal(size = (ncols, nfacts))
    cov = np.dot(w, w.T) # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size = ncols)) # full rank cov
    return cov

def cov2Corr(cov): # 
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1 # numerical error
    return corr

def errorPDFs(var, eval, q, bwidth, pts = 1000):
    # Fit error
    pdf0 = marcenkoPasturPDF(var, q, pts) # theoretical pdf
    pdf1 = fitKDE(eval, bwidth, x = pdf0.index.values) # empirical pdf
    sse = np.sum((pdf1 - pdf0)**2)
    return sse

def findMaxEval(eval, q, bwidth):
    # Find max random eval by fitting Marcenko’s dist
    out = minimize(lambda *x:errorPDFs(*x), .5, args = (eval, q, bwidth), bounds = ((1E-5 , 1-1E-5), ))
    if out['success']:var = out['x'][0]
    else:var = 1
    emax = var*(1 + (1./q)**.5)**2
    return emax, var

def denoisedCorr(eval, #
                evec, #
                nfacts): #
    # Remove noise from corr by fixing random eigenvalues
    eval_ = np.diag(eval).copy()
    eval_[nfacts:] = eval_[nfacts:].sum()/float(eval_.shape[0] - nfacts)
    eval_= np.diag(eval_)
    corr1 = np.dot(evec,eval_).dot(evec.T)
    corr1 = cov2Corr(corr1)
    return corr1

def denoisedCorr2(eval, #
                  evec, # 
                  nfacts, # 
                  alpha = 0): # 
    # Remove noise from corr through targeted shrinkage
    evalL, evecL = eval[:nfacts, :nfacts], evec[:, :nfacts]
    evalR, evecR = eval[nfacts:, nfacts:], evec[:, nfacts:]
    corr0 = np.dot(evecL,evalL).dot(evecL.T)
    corr1 = np.dot(evecR,evalR).dot(evecR.T)
    corr2 = corr0 + alpha*corr1 + (1 - alpha)*np.diag(np.diag(corr1))
    return corr2

def formBlockMatrix(nblocks, #
                    bsize, #
                    bcorr): #
    block = np.ones((bsize,bsize))*bcorr
    block[range(bsize), range(bsize)] = 1
    corr = block_diag(*([block]*nblocks))
    return corr

def formTrueMatrix(nblocks,bsize,bcorr):
    corr0 = formBlockMatrix(nblocks, bsize, bcorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep = True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2Cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1,1)
    return mu0, cov0

def simCovMu(mu0, #
            cov0, #
            nObs, #
            shrink = False): #
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size = nObs)
    mu1 = x.mean(axis = 0).reshape(-1,1)
    if shrink:cov1 = LedoitWolf().fit(x).covariance_
    else:cov1 = np.cov(x, rowvar = 0)
    return mu1, cov1

def corr2Cov(corr, #
            std): #
    cov = corr*np.outer(std, std)
    return cov

def deNoiseCov(cov0, # 
                q, #
                bwidth): #
    corr0 = cov2Corr(cov0) # 
    eval0, evec0 = PCA(corr0)
    emax0, var0 = findMaxEval(np.diag(eval0), q, bwidth)
    nfacts0 = eval0.shape[0] - np.diag(eval0)[::-1].searchsorted(emax0)
    corr1 = denoisedCorr(eval0, evec0, nfacts0)
    cov1 = corr2Cov(corr1, np.diag(cov0)**.5)
    return cov1

def optPort(cov, # 
            mu = None): # 
    inv = np.linalg.inv(cov)
    ones = np.ones(shape = (inv.shape[0], 1))
    if mu is None:mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w