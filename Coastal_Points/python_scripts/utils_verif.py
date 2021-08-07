from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

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
import xarray as xr

import matplotlib
#mapping
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import Unet_b
import utils_CNN
from tensorflow.python.client import device_lib
import utilsAnEn
from scipy.stats import norm
import time
import random 
from tqdm import tqdm


# file switch: 
def load_forecast(fcast,whichone):
    
    if whichone not in ['Raw_gefs','MV_gefs','NN_gefs','NN_ref','CNN_ref','CNNft_ref','AnEn','Reforecast','NNft_ref','EMOS']:
        raise NameError('forecast must be one of : Raw_gefs, MV_gefs , NN_ref , CNN_ref , CNNft_ref , AnEn, NNft_ref, EMOS, or Reforecast ')
    
    
    if whichone == 'Raw_gefs':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/GEFS_out/RAW_gefs.pkl')
        Pre_m = np.array(df['IVTmean'])
        Pre_s = np.array(df['IVTstd'])
        Post_m = np.array(df['IVTmean'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
        print('No Post-Processing')
        
    if whichone == 'MV_gefs':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/GEFS_out/Mean_Variance_PP_gefs.pkl')
        Pre_m = np.array(df['IVTmean'])
        Pre_s = np.array(df['IVTstd'])
        Post_m = np.array(df['IVTmean'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
        
    if whichone == 'NN_gefs':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/GEFS_out/NN_CRPS_PP_gefs.pkl')
        Pre_m = np.array(df['Modelmean'])
        Pre_s = np.array(df['Modelstd'])
        Post_m = np.array(df['IVTmean'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
        
    if whichone == 'NN_ref':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/Reforecast_out/NN_CRPS_PP_ref2.pkl')
        Pre_m = np.array(df['Model'])
        Pre_s = np.zeros(len(Pre_m))
        Post_m = np.array(df['IVTmean'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
        
    if whichone == 'NNft_ref':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/Reforecast_out/NN_FineTune_CRPS_PP_ref2.pkl')
        Pre_m = np.array(df['Model'])
        Pre_s = np.zeros(len(Pre_m))
        Post_m = np.array(df['IVTmean'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
        
    if whichone == 'CNN_ref':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/Reforecast_out/NN_CRPS_CNNPP_ref2.pkl')
        Pre_m = np.array(df['Model'])
        Pre_s = np.zeros(len(Pre_m))
        Post_m = np.array(df['IVTmean'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
    
    if whichone == 'CNNft_ref':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/Reforecast_out/NN_CRPS_CNNPP_FINETUNEref2.pkl')
        Pre_m = np.array(df['Model'])
        Pre_s = np.zeros(len(Pre_m))
        Post_m = np.array(df['IVTmean'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
        
    if whichone == 'AnEn':
        df = utilsAnEn.load_picks([2016,2017,2018],fcast,21)
        Pre_m = np.array(df['Model'])
        Pre_s = np.zeros(len(Pre_m))
        Post_m = np.array(np.mean(df.filter(regex='Analog'),axis=1))
        Post_s = np.array(np.std(df.filter(regex='Analog'),axis=1))
        Obs = np.array(df['OBS'])
        
        
    if whichone == 'Reforecast':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/Reforecast_out/NN_CRPS_CNNPP_ref2.pkl')
        Pre_m = np.array(df['Model'])
        Pre_s = np.zeros(len(Pre_m))
        Post_m = np.array(df['Model'])
        Post_s = np.zeros(len(Pre_m))
        Obs = np.array(df['OBS'])
        
    if whichone == 'EMOS':
        df = pd.read_pickle('/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/Reforecast_out/FCN_FINETUNE_CRPS_PP_ref2.pkl')
        Pre_m = np.array(df['Model'])
        Pre_s = np.zeros(len(Pre_m))
        Post_m = np.array(df['Model'])
        Post_s = np.array(df['IVTstd'])
        Obs = np.array(df['OBS'])
        

    return df, Pre_m, Pre_s, Post_m, Post_s, Obs






def load_forecast_concat(fcasts,whichone): 
    
    if whichone not in ['Raw_gefs','MV_gefs','NN_gefs','NN_ref','CNN_ref','CNNft_ref','AnEn','Reforecast','NNft_ref','EMOS']:
        raise NameError('forecast must be one of : Raw_gefs, MV_gefs , NN_ref , CNN_ref , CNNft_ref , AnEn, NNft_ref, EMOS, or Reforecast ')
        
    for ind,fcast in enumerate(fcasts):
        dft, Pre_mt, Pre_st, Post_mt, Post_st, Obst = load_forecast(fcast,whichone)
        
        if ind == 0:
            df = dft
            Pre_m = Pre_mt
            Pre_s = Pre_st
            Post_m = Post_mt
            Post_s = Post_st
            Obs = Obst
        else:
            df = pd.concat([df,dft]).reset_index(drop=True)
            Pre_m = np.concatenate([Pre_m,Pre_mt])
            Pre_s = np.concatenate([Pre_s,Pre_st])
            Post_m = np.concatenate([Post_m,Post_mt])
            Post_s = np.concatenate([Post_s,Post_st])
            Obs = np.concatenate([Obs,Obst])
    return df, Pre_m, Pre_s, Post_m, Post_s, Obs

#probability of value given mean and std. 
def prob_ob(x,u,s):
    z = (x-u)/s
    return 1-norm.cdf(z)

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
    mu = y_pred[:,:,:,0]
    sigma = K.abs(y_pred[:,:,:,1])
    # Ugly workaround for different tensor allocation in keras and theano
    if not theano:
        y_true = y_true[:,:,:,0]   # Need to also get rid of axis 1 to match!

    # To stop sigma from becoming negative we first have to 
    # convert it the the variance and then take the square
    # root again. 
    var = K.square(sigma)
    # The following three variables are just for convenience
    loc = (y_true - mu) / sigma
    phi = 1.0 / np.sqrt(2.0 * np.pi) * K.exp(-K.square(loc) / 2.0)
    Phi = 0.5 * (1.0 + tfm.erf(loc / np.sqrt(2.0)))
    # First we will compute the crps for each input/target pair
    crps =  sigma * (loc * (2. * Phi - 1.) + 2 * phi - 1. / np.sqrt(np.pi))
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


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def f_emp(ens,x):
    xs,ys = ecdf(ens)
    vals=[]
    for xT in x:
        if all(xs>xT):
            vals.append(0)
        elif (np.sum(1*(xs>xT))>0) & (np.sum(1*(xs>xT))<len(xs)):
            inds = np.searchsorted(xs>xT, 0.5)
            vals.append(ys[inds-1])
        else:
            vals.append(1)
    return np.array(vals)

### computation of threshold-weighted CRPS via numerical approximation suggested by Gneiting & Ranjan (2011)
def twcrps_sample(y,ens,weight,thr,step_width=0.001):
    #y = real-valued observation (numpy array)
    #ens = vector of ensemble forecasts (numpy array)
    #thr = threshold parameter for weight function
    # weight = charactor vector identifying the employed weight function:
    #     "indicator.right": Indicator weight function for right tail
    #     "indicator.left": Indicator weight function for left tail
    #     "normalCDF.right": Gaussian weight function for right tail, mean = thr, and sd = 1
    #     "normalCDF.left": Gaussian weight function for right tail, mean = thr, and sd = 1
    # lower, upper: bounds for interval of evaluation points
    # step.width: increment of the sequence of evaluation points

    lower = np.min(np.append(ens,y))
    upper = np.max(np.append(ens,y))
    eval_z = np.arange(lower,upper,step_width)
    if weight not in ['indicator.right','indicator.left','normalCDF.right','normalCDF.left']:
        raise ValueError("weight function not specified correctly")
        
    if weight =='indicator.right':
        def w(z):return z>=thr
    elif weight =='indicator.left':
        def w(z):return z<=thr
    elif weight == 'normalCDF.right':
        def w(z):return stats.norm.cdf(z, loc=thr, scale=1)
    elif weight == 'normalCDF.left':
        def w(z):return 1-stats.norm.cdf(z, loc=thr, scale=1)

    out = np.sum(w(eval_z)*((f_emp(ens,eval_z)-(y<=eval_z)))**2)*(upper-lower)/len(eval_z)
    return out




def reliability_diagrams(predictions, truths, confidences, bin_size=0.1, n_boot=1000):

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    accs = []
    o_f = []
    # Compute empirical probability for each bin
    plot_x = []
    for conf_thresh in upper_bounds:
        acc, perc_pred, avg_conf,obs_freq = compute_accuracy(conf_thresh-bin_size, conf_thresh, confidences, predictions, truths)
        plot_x.append(avg_conf)
        accs.append(acc)
        o_f.append(obs_freq)

    # Produce error bars for each bin
    upper_bound_to_bootstrap_est = {x:[] for x in upper_bounds}
    for i in tqdm(range(n_boot)):

        # Generate bootstrap
        boot_strap_outcomes = []
        boot_strap_confs = random.sample(list(confidences), len(confidences))
        for samp_conf in boot_strap_confs:
            correct = 0
            if random.random() < samp_conf:
                correct = 1
            boot_strap_outcomes.append(correct)

        # Compute error frequency in each bin
        for upper_bound in upper_bounds:
            conf_thresh_upper = upper_bound
            conf_thresh_lower = upper_bound - bin_size

            filtered_tuples = [x for x in zip(boot_strap_outcomes, boot_strap_confs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
            correct = len([x for x in filtered_tuples if x[0] == 1])
            acc = float(correct) / len(filtered_tuples) if len(filtered_tuples) > 0 else 0

            upper_bound_to_bootstrap_est[upper_bound].append(acc)
       
    upper_bound_to_bootstrap_upper_bar = {}
    upper_bound_to_bootstrap_lower_bar = {}
    for upper_bound, freqs in upper_bound_to_bootstrap_est.items():
        top_95_quintile_i = int(0.975 * len(freqs))
        lower_5_quintile_i = int(0.025 * len(freqs))

        upper_bar = sorted(freqs)[top_95_quintile_i]
        lower_bar = sorted(freqs)[lower_5_quintile_i]

        upper_bound_to_bootstrap_upper_bar[upper_bound] = upper_bar
        upper_bound_to_bootstrap_lower_bar[upper_bound] = lower_bar

    upper_bars = []
    lower_bars = []
    for i, upper_bound in enumerate(upper_bounds):
        if upper_bound_to_bootstrap_upper_bar[upper_bound] == 0:
            upper_bars.append(0)
            lower_bars.append(0)
        else:
            # The error bar arguments need to be the distance from the data point, not the y-value
            upper_bars.append(abs(plot_x[i] - upper_bound_to_bootstrap_upper_bar[upper_bound]))
            lower_bars.append(abs(plot_x[i] - upper_bound_to_bootstrap_lower_bar[upper_bound]))
    new_plot_x = []
    new_accs = []
    for i, bars in enumerate(zip(lower_bars, upper_bars)):
        if bars[0] == 0 and bars[1] == 0:
            continue
        new_plot_x.append(plot_x[i])
        new_accs.append(o_f[i])
    
    return plot_x,new_plot_x, new_accs,lower_bars,upper_bars




def compute_accuracy(conf_thresh_lower, conf_thresh_upper, conf, pred, true):

    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
        accuracy = float(correct)/len(filtered_tuples)
        perc_of_data = float(len(filtered_tuples))/len(conf)
        obs_freq = len([x for x in filtered_tuples if x[1] != 0])/float(len(filtered_tuples))
        return accuracy, perc_of_data, avg_conf,obs_freq
    
def freq_hist(confidences,bin_size=0.1):
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    accs = []
    o_f = []
    # Compute density per bin. 
    plot_x = []
    for conf_thresh in upper_bounds:
        nummy=0
        beans = [x for x in confidences if x > (conf_thresh-bin_size) and x <= conf_thresh]
        plot_x.append(len(beans))
    return plot_x
        
    
def brier_decompose(verif_for,probs_for):
    binwidth=0.05
    obs_mean = np.mean(verif_for)
    bsres = np.nan * np.zeros(len(verif_for), 'float')
    bsrel = np.nan * np.zeros(len(verif_for), 'float')
    edges =np.arange(0,1+binwidth,binwidth)
    edges[-1]=1.02
    for i in range(0, len(edges) - 1):
        I = np.where((probs_for >= edges[i]) & (probs_for < edges[i + 1]))[0]
        if len(I)>0:
            obs_mean_I = np.mean(verif_for[I])
            bsres[I] = (obs_mean_I-obs_mean)**2
            bsrel[I] = (probs_for[I]-obs_mean_I)**2
    bsres = np.nanmean(bsres)
    bsrel = np.nanmean(bsrel)
    bsunc = np.nanmean((obs_mean-verif_for)**2)
    bsS = np.sum((verif_for-probs_for)**2)/len(verif_for)
    return bsS,bsres,bsrel,bsunc,bsrel - bsres + bsunc