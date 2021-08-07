####################################################################################
### Import Packages ### run in tfp environment: 
####################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.keras.optimizer_v2.adam import Adam
tfd = tfp.distributions
import tensorflow.keras.backend as K
from tensorflow import math as tfm

import os

import utilsProb
import utilsProbSS
import glob
import sys
from scipy.stats import rankdata
import pandas as pd
import importlib
import copy
from netCDF4 import Dataset, num2date
from scipy.interpolate import interpn
from matplotlib.colors import Normalize 
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
import properscoring as ps
from math import erf


import matplotlib
#mapping
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import xarray as xr

####################################################################################
####################################################################################

negloglik = lambda y, rv_y: -rv_y.log_prob(y)


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


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype,name='banjo'),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1),name='banjo2'),
  ])


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype,name='spoon'),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1),name='doon'),
  ])



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


def crps_cost_function_LogNorm(y_true, y_pred, theano=False):
    """Compute the CRPS cost function for a lognormal distribution defined by
    the mean and standard deviation.
    Big Ups to Stephan Rasp & Sebastian Lerch. 
    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
        theano: Set to true if using this with pure theano.
    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    y = y_true
    # Ugly workaround for different tensor allocation in keras and theano
    if not theano:
        y= y_true[:, 0]   # Need to also get rid of axis 1 to match!

    # To stop sigma from becoming negative we first have to 
    # convert it the the variance and then take the square
    # root again. 
    sigma = K.abs(sigma)
    var = K.square(sigma)
    # The following three variables are just for convenience
    loc = (y_true - mu) / sigma
    
    c1 = y * (2* 0.5 + 0.5*(tfm.erf((tfm.log(y)-mu)/(np.sqrt(2)*sigma))))
    
    c2 = 2*K.exp(mu+0.5*K.square(sigma))
    
    c3 = 0.5 + 0.5*(tfm.erf((tfm.log(y)-(mu+sigma**2))/(np.sqrt(2)*sigma)))
    
    c4 = (0.5*(1+tfm.erf((sigma/np.sqrt(2))/np.sqrt(2))))-1
    
    # First we will compute the crps for each input/target pair
    crps = c1-c2*(c3+c4)
        
    # Then we take the mean. The cost is now a scalar
    diff = y-(K.exp(mu+((K.square(sigma))/2)))
    greater = K.greater(diff,0)
    greater = K.cast(greater, K.floatx())*1 #0 for lower, 1 for greater
#     greater = greater + 1                 #1 for lower, 5 for greater

    return K.mean(crps) #+ K.mean(greater*diff)

def crps_logis_cost_function(y_true, y_pred, theano=False):
    """Compute the CRPS cost function for a logistic distribution defined by
    the mean and standard deviation.
    Big Ups to Stephan Rasp & Sebastian Lerch. 
    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
        theano: Set to true if using this with pure theano.
    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    y = y_true
    # Ugly workaround for different tensor allocation in keras and theano
    if not theano:
        y= y_true[:, 0]   # Need to also get rid of axis 1 to match!

    # To stop sigma from becoming negative we first have to 
    # convert it the the variance and then take the square
    # root again. 
    sigma = K.abs(sigma)
    var = K.square(sigma)
    # The following three variables are just for convenience
    loc = (y_true - mu) / sigma
    
    
    # First we will compute the crps for each input/target pair
    crps =  K.sqrt(var) * (loc-2*K.log(1/(1+K.exp(-loc)))-1)
    # Then we take the mean. The cost is now a scalar
    return K.mean(crps)


def crps_cost_function(y_true, y_pred, theano=False):
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
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    # Ugly workaround for different tensor allocation in keras and theano
    if not theano:
        y_true = y_true[:, 0]   # Need to also get rid of axis 1 to match!

    # To stop sigma from becoming negative we first have to 
    # convert it the the variance and then take the square
    # root again. 
    var = K.square(sigma)
    # The following three variables are just for convenience
    loc = (y_true - mu) / K.sqrt(var)
    phi = 1.0 / np.sqrt(2.0 * np.pi) * K.exp(-K.square(loc) / 2.0)
    Phi = 0.5 * (1.0 + tfm.erf(loc / np.sqrt(2.0)))
    # First we will compute the crps for each input/target pair
    crps =  K.sqrt(var) * (loc * (2. * Phi - 1.) + 2 * phi - 1. / np.sqrt(np.pi))
    # Then we take the mean. The cost is now a scalar
    return K.mean(crps)


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
    Phi = 0.5 * (1.0 + math.erf(loc / np.sqrt(2.0)))
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



def build_emb_model(n_features, n_outputs, hidden_nodes, emb_size, max_id,
                    compile=False, optimizer='adam', lr=0.0001,
                    loss=crps_cost_function,
                    activation='relu', reg=None):
    """
    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        hidden_nodes: int or list of hidden nodes
        emb_size: Embedding size
        max_id: Max embedding ID
        compile: If true, compile model
        optimizer: Name of optimizer
        lr: learning rate
        loss: loss function
        activation: Activation function for hidden layer
    Returns:
        model: Keras model
    """
    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]

    features_in = tf.keras.layers.Input(shape=(n_features,))
    id_in = tf.keras.layers.Input(shape=(1,))
    emb = tf.keras.layers.Embedding(max_id + 1, emb_size)(id_in)
    emb = tf.keras.layers.Flatten()(emb)
    x = tf.keras.layers.Concatenate()([features_in, emb])
    for h in hidden_nodes:
        x = tf.keras.layers.Dense(h, activation=activation, kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dense(n_outputs, activation='linear', kernel_regularizer=reg)(x)
    model = tf.keras.models.Model(inputs=[features_in, id_in], outputs=x)

    if compile:
        opt = tf.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss)
    return model




def build_model(n_features, n_outputs, hidden_nodes,
                    compile=False, optimizer='adam', lr=0.0001,
                    loss=crps_cost_function,
                    activation='relu', reg=None):
    """
    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        hidden_nodes: int or list of hidden nodes
        emb_size: Embedding size
        max_id: Max embedding ID
        compile: If true, compile model
        optimizer: Name of optimizer
        lr: learning rate
        loss: loss function
        activation: Activation function for hidden layer
    Returns:
        model: Keras model
    """
    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]

    features_in = tf.keras.layers.Input(shape=(n_features,))
    id_in = tf.keras.layers.Input(shape=(1,))

    x = features_in
    for h in hidden_nodes:
        x = tf.keras.layers.Dense(h, activation=activation, kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dense(n_outputs, activation='linear', kernel_regularizer=reg)(x)
    model = tf.keras.models.Model(inputs=features_in, outputs=x)

    if compile:
        opt = tf.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss)
    return model






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