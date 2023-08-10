
import numpy as np,scipy.stats as ss
import math
from sklearn.metrics import mutual_info_score

"""----------------------------------------------------------------------
function: Calculates Variation of Information
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.2, Page 44
----------------------------------------------------------------------"""
def variationsInformation(x,# first data
                          y,# second data
                          bins, # number of bins
                          norm = False): # for normalized
    # variation of information
    histogramXY = np.histogram2d(x, y, bins)[0] # hist2d of x, y
    mutualInformation = mutual_info_score(None, None, contingency = histogramXY)  # mutual score from hist2d
    marginalX = ss.entropy(np.histogram(x, bins)[0]) # marginal
    marginalY = ss.entropy(np.histogram(y, bins)[0]) # marginal
    variationXY = marginalX + marginalY - 2*mutualInformation # variation of information
    if norm:
        jointXY = marginalX + marginalY - mutualInformation # joint
        variationXY /= jointXY # normalized variation of information
    return variationXY

"""----------------------------------------------------------------------
function: Calculates number of bins for histogram
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.3, Page 46
----------------------------------------------------------------------"""
def numberBins(numberObservations, # number of obs
               correlation = None): # corr
    # Optimal number of bins for discretization
    if correlation is None: # univariate case
        z = (8 + 324*numberObservations + 12*(36*numberObservations + 729*numberObservations**2)**.5)**(1/3.)
        bins = round(z/6. + 2./(3*z) + 1./3) # bins
    else: # bivariate case
        bins = round(2**-.5*(1 + (1 + 24*numberObservations/(1. - correlation**2))**.5)**.5)  # bins
    return int(bins)

"""----------------------------------------------------------------------
function: Calculates Variation of Information with calculating number of bins
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.3, Page 46
----------------------------------------------------------------------"""
def variationsInformationExtended(x, # data1
                                  y, # data2
                                  norm = False): # for normalized variations of info
    # variation of information
    numberOfBins = numberBins(x.shape[0], correlation = np.corrcoef(x, y)[0, 1])# calculate number of bins
    histogramXY = np.histogram2d(x, y, numberOfBins)[0] # hist2d of x,y
    mutualInformation = mutual_info_score(None, None, contingency = histogramXY) # mutual score
    marginalX = ss.entropy(np.histogram(x, numberOfBins)[0]) # marginal
    marginalY = ss.entropy(np.histogram(y, numberOfBins)[0]) # marginal
    variationXY = marginalX + marginalY - 2*mutualInformation # variation of information
    if norm:
        jointXY = marginalX + marginalY - mutualInformation # joint
        variationXY /= jointXY # normalized variation of information
    return variationXY

"""----------------------------------------------------------------------
function: Calculates Mutual  Information with calculating number of bins
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.4, Page 48
----------------------------------------------------------------------"""
def mutualInformation(x, # data1
                      y, #data2
                      norm = False): #for normalized mutual info
    # mutual information
    numberOfBins = numberBins(x.shape[0], correlation = np.corrcoef(x, y)[0, 1]) # calculate number of bins
    histogramXY = np.histogram2d(x, y, numberOfBins)[0] # hist2d of x,y
    mutualInformation = mutual_info_score(None, None, contingency = histogramXY) # mutual score
    if norm:
        marginalX = ss.entropy(np.histogram(x, numberOfBins)[0]) # marginal
        marginalY = ss.entropy(np.histogram(y, numberOfBins)[0]) # marginal
        mutualInformation /= min(marginalX, marginalY) # normalized mutual information
    return mutualInformation

"""----------------------------------------------------------------------
function: Calculates distance from a dependence matrix
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
def distance(dependence,  # dependence matrix
             metric = "angular"): # method
    if metric == "angular":
        distance = ((1 - dependence).round(5)/2.)**0.5
    elif metric == "absolute_angular":
        distance = ((1 - abs(dependence)).round(5)/2.)**0.5
    return distance

"""----------------------------------------------------------------------
function: Calculates KullbackLeibler divergence from two discrete probability distributions defined on the same probability space
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
def KullbackLeibler(p, # first distribution
                    q): # second distribution
    divergence = -(p*math.log(q/p)).sum() # calculate divenrgence
    return divergence

"""----------------------------------------------------------------------
function: Calculates crossentropy from two discrete probability distributions defined on the same probability space
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
def crossEntropy(p,  # first distribution
                 q): # second distribution
    entropy = -(p*math.log(q)).sum() # calculate entropy
    return entropy