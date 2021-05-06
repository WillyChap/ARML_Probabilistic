from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import os
import glob
import sys
from scipy.stats import rankdata
import pandas as pd
import copy
from scipy.interpolate import interpn

import matplotlib as mpl
import seaborn as sns
sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
from math import erf


def load_picks(yeahz,LT,Ens):
    for cnt,bb in enumerate(yeahz):
        print(bb)
        if cnt==0:
            yago = pd.read_pickle('./AnEn/Analog_Total_Bias_300_21_'+ str( bb) +'_21Ense_startingyear_1985.pkl')
            yago = yago[yago['lead_hr']==LT]
        else: 
            Tago = pd.read_pickle('./AnEn/Analog_Total_Bias_300_21_'+ str( bb) +'_21Ense_startingyear_1985.pkl')
            Tago = Tago[Tago['lead_hr']==LT]
            yago = pd.concat([yago,Tago],axis=0)
            
        totter = yago.filter(regex='Analog').shape[1]
        
        if Ens != totter:
            dropcols = ['Analog'+f'{n:1}' for n in range(Ens,totter) ]
            yago = yago.drop(columns=dropcols)
            
    return yago


def load_picks_IVTonly(yeahz,LT,Ens):
    for cnt,bb in enumerate(yeahz):
        print(bb)
        if cnt==0:
            print('IVT ONLY!')
            yago = pd.read_pickle('./AnEn/Analog_Total_Bias_300_100_' + str( bb) +'_IVTonly_startingyear_1985.pkl')
            yago = yago[yago['lead_hr']==LT]
        else: 
            Tago = pd.read_pickle('./AnEn/Analog_Total_Bias_300_100_' + str( bb) +'_IVTonly_startingyear_1985.pkl')
            Tago = Tago[Tago['lead_hr']==LT]
            yago = pd.concat([yago,Tago],axis=0)
            
        totter = yago.filter(regex='Analog').shape[1]
        
        if Ens != totter:
            dropcols = ['Analog'+f'{n:1}' for n in range(Ens,totter) ]
            yago = yago.drop(columns=dropcols)
            
    return yago

def load_picks_year(yeahz,LT,Ens,yrtrain):
    for cnt,bb in enumerate(yeahz):
        print(bb)
        print('./AnEn/Analog_Total_Bias_300_100_' + str( bb) +'_startingyear_'+str(yrtrain)+'.pkl')
        if cnt==0:
            yago = pd.read_pickle('./AnEn/Analog_Total_Bias_300_100_' + str( bb) +'_startingyear_'+str(yrtrain)+'.pkl')
            yago = yago[yago['lead_hr']==LT]
        else: 
            Tago = pd.read_pickle('./AnEn/Analog_Total_Bias_300_100_' + str( bb) +'_startingyear_'+str(yrtrain)+'.pkl')
            Tago = Tago[Tago['lead_hr']==LT]
            yago = pd.concat([yago,Tago],axis=0)
            
        totter = yago.filter(regex='Analog').shape[1]
        
        if Ens != totter:
            dropcols = ['Analog'+f'{n:1}' for n in range(Ens,totter) ]
            yago = yago.drop(columns=dropcols)
            
    return yago


def cdf_of_normal(true,ypred,scale):
    return 0.5*(1+erf((true-ypred)/(scale*np.sqrt(2))))

def rmse(guess,truth):
    n = len(truth)
    rms = np.linalg.norm(guess - truth) / np.sqrt(n)
    return rms


def bias(guess,truth):
    bb = np.mean(guess)-np.mean(truth)
    return bb

def corrss(guess,truth):
    bb = np.corrcoef(np.squeeze(guess),np.squeeze(truth))[0,1]
    return bb


def crmse(guess,truth):
    guess = np.squeeze(guess)
    truth = np.squeeze(truth)
    mg = np.mean(guess)
    mt = np.mean(truth)
    n = len(truth)
    bb=np.linalg.norm((guess-mg) - (truth-mt)) / np.sqrt(n)
    return bb

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def spreadskill(guess,truth,stddevver,numbins):
#     guess: postprocess guess any input shape
#     truth: true value any input shape 
#     stddevver: standard deviation of the guess any input shape
#     returns: 
#         variance mean, mse mean, bootstraped variance, bootrapped mean
    
    distPP = np.ndarray.flatten(stddevver)
    guess = np.ndarray.flatten(guess)
    truth = np.ndarray.flatten(truth)
    
    indexsort = np.argsort(distPP)
    err = (guess- truth)**2
    varianceall = distPP[indexsort]
    err_sort = err[indexsort]

    numbins = numbins 
    inds = np.zeros(len(varianceall))
    numst = 0
    for nn in range(numbins):
        numdo = int(len(varianceall)/numbins)
        inds[numst:numdo*(nn+1)] = nn
    
        if nn ==np.max(range(numbins)):
            inds[numst:] = nn
        numst+=numdo
    
    avgvar_m = []
    msebin_m = []

    avgvar_s = []
    msebin_s = []
    nummy =2000

    for bb in np.unique(inds):
        locbin = np.where([inds==bb])[1]
        tavg =0
        tmse =0
        tavg = np.zeros(nummy)
        tmse = np.zeros(nummy)
        for ii in range(nummy):
            tavg[ii]  = np.mean(np.random.choice(varianceall[locbin],len(locbin)))
            tmse[ii]  = np.sqrt(np.mean(np.random.choice(err_sort[locbin],len(locbin))))
        avgvar_m = np.append(avgvar_m,np.mean(tavg))
        msebin_m = np.append(msebin_m,np.mean(tmse))
        avgvar_s = np.append(avgvar_s,np.std(tavg))
        msebin_s = np.append(msebin_s,np.std(tmse))
    return avgvar_m,msebin_m,avgvar_s,msebin_s

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots(figsize=(10, 8))
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


def stnd_error_mean(guess,truth,numbins):
#     guess: postprocess guess any input shape
#     truth: true value any input shape 
#     stddevver: standard deviation of the guess any input shape
#     returns: 
#         variance mean, mse mean, bootstraped variance, bootrapped mean
    
    guess = np.ndarray.flatten(guess)
    truth = np.ndarray.flatten(truth)
    

    err = (guess- truth)**2
    indexsort = np.argsort(err)
    err_sort = err[indexsort]
    
    truthSORT = truth[indexsort]
    guessSORT = guess[indexsort]

    numbins = numbins 
    inds = np.zeros(len(err_sort))
    numst = 0
    for nn in range(numbins):
        numdo = int(len(err_sort)/numbins)
        inds[numst:numdo*(nn+1)] = nn
    
        if nn ==np.max(range(numbins)):
            inds[numst:] = nn
        numst+=numdo
        
    msebin_m = []
    msebin_s = []
    nummy =2000
    maxlist=[]
    for bb in np.unique(inds):
        locbin = np.where([inds==bb])[1]
        tmse =0
        tmse = np.zeros(nummy)
        for ii in range(nummy):
            tmse[ii]  = np.sqrt(np.mean(np.random.choice(err_sort[locbin],len(locbin))))
        msebin_m = np.append(msebin_m,np.mean(tmse))
        msebin_s = np.append(msebin_s,np.std(tmse))
        
        if bb == np.max(np.unique(inds)):
            print('...getting max...')
            mlt = truthSORT[locbin]
            mlg = guessSORT[locbin]
            mle = np.sqrt((truthSORT[locbin]-guessSORT[locbin])**2)
            
        
    return msebin_m,msebin_s,mlt,mlg,mle






#MAKE SURE YOUR true values are NEVER zero when using this. 


def crps_Lnorm_TEST(y_true, y_pred, sig_pred):
    """Compute the CRPS cost function for a normal distribution defined by
    the mean and standard deviation.
    Code inspired by Kai Polsterer (HITS).
    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """
    # Split input
    mu = y_pred
    sigma = sig_pred
    y = y_true
    c1 = y * (2*plnorm(y,mu,sigma)-1)
    c2 = 2*np.exp(mu+0.5*sigma**2)
    c3 = plnorm(y,mu+sigma**2,sigma) 
    c4 = pnorm(sigma/np.sqrt(2))-1
    crps = c1-c2*(c3+c4)
    return np.mean(crps)


def crps_logis_TEST(y_true, y_pred, sig_pred):
    """Compute the CRPS cost function for a normal distribution defined by
    the mean and standard deviation.
    Code inspired by Kai Polsterer (HITS).
    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
        theano: Set to true if using this with pure theano.
    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred
    sigma = sig_pred
    var = sigma**2
    
    # The following three variables are just for convenience
    loc = (y_true - mu) / np.sqrt(var)
    # First we will compute the crps for each input/target pair
    crps =  np.sqrt(var) * (loc-2*np.log(1/(1+np.exp(-loc)))-1)
    # Then we take the mean. The cost is now a scalar
    return np.mean(crps)


def crps_TEST(y_true, y_pred, sig_pred):
    """Compute the CRPS cost function for a normal distribution defined by
    the mean and standard deviation.
    Code inspired by Kai Polsterer (HITS).
    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
        sig_pred: sigma of prediction (normal). 
        theano: Set to true if using this with pure theano.
    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred
    sigma = sig_pred

    # To stop sigma from becoming negative we first have to 
    # convert it the the variance and then take the square
    # root again. 
    var = sigma**2
    # The following three variables are just for convenience
    loc = (y_true - mu) / np.sqrt(var)
    phi = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(loc**2) / 2.0)
    Phi = 0.5 * (1.0 + erf(loc / np.sqrt(2.0)))
    # First we will compute the crps for each input/target pair
    crps =  np.sqrt(var) * (loc * (2. * Phi - 1.) + 2 * phi - 1. / np.sqrt(np.pi))
    # Then we take the mean. The cost is now a scalar
    return np.mean(crps)

def plnorm(y,mu,sigma):
    nummy=0.5 + 0.5*(math.erf((np.log(y)-mu)/(np.sqrt(2)*sigma)))
    return nummy 

def pnorm(loc):
    nummy=0.5*(1+math.erf(loc/np.sqrt(2)))
    return nummy 


def lognstat(mu, sigma):
    """Calculate the mean of and variance of the lognormal distribution given
    the mean (`mu`) and standard deviation (`sigma`), of the associated normal 
    distribution."""
    m = np.exp(mu + sigma**2 / 2.0)
    v = np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1)
    return m, v


def ranker(obs_array,Ensemble):
    """Compute the rank histogram rankings
    obs_array = np.array[time,];           e.g. obs_array.shape = (136884,)
    Ensemble = np.array([Ensemble,time]);  e.g. Ensemble.shape  = (15, 136884)
     """
    combined=np.vstack((np.array(obs_array)[np.newaxis],Ensemble))
    print('computing ranks')
    ranks=np.apply_along_axis(lambda x: rankdata(x,method='min'),0,combined)

    print('computing ties')
    ties=np.sum(ranks[0]==ranks[1:], axis=0)
    ranks=ranks[0]
    
    return ranks