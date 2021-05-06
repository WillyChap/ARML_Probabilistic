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
import utilsProbSS_Ens
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

import comsnngefs
####################################################################################
### Import Packages ### run in tfp environment: 
####################################################################################
####################################################################################
####################################################################################
####################################################################################

####################################################################################
####################################################################################
stepnum=int(sys.argv[1]) #example input 'F048'
####################################################################################
####################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

####################################################################################
####################################################################################

dd = '/glade/scratch/wchapman/GEFS/'
os.chdir(dd)

yago= next(os.walk('.'))[1]
yago = sorted(yago)
subs = 'F'
res = [ii for ii in yago if subs in ii] 
res
for fcast in res[:stepnum]: 
    os.chdir(dd+'/'+fcast)
    print(os.getcwd())
    
####################################################################################
####################################################################################

batch_num = 50
epochs = 30
#find all files in directory                                                                                                                                                  
print('Training on')
path = os.getcwd()
train_file_names = sorted([f for f in glob.glob(path + "/*Clean_noEnse*.nc", recursive=True)])
for f in train_file_names:
    print(f)
All_file_names =train_file_names

####################################################################################
####################################################################################

rang = 144
latlonfolder = '/glade/scratch/wchapman/AnEnCNN_good/Data/WestCoast/'
[latsDO,lonsDO,latind, lonind] = utilsProbSS.get_latlon_ind_gefs(latlonfolder)
latsDO.shape
latind = np.array(latind[:rang])
lonind = np.array(lonind[:rang])
latsDO = np.array(latsDO[:rang])
lonsDO = np.array(lonsDO[:rang])

norm_dict = utilsProbSS_Ens.get_image_normalization_params(All_file_names,latind,lonind)
sys.stdout.flush()
norm_dict_targ = utilsProbSS_Ens.get_image_normalization_params_targ(All_file_names,latind,lonind)

# sys.stdout.flush()
num_samps_train = utilsProbSS_Ens.count_samps(All_file_names,latind,lonind)
# num_samps_val = utilsProb.count_samps(validate_file_names)
# num_samps_test = utilsProb.count_samps(test_file_names)

print('number of training samples:',num_samps_train)
# sys.stdout.flush()
num_samps_tot = num_samps_train

norm_dict_targ['IVTm'][0]=0
norm_dict_targ['IVTm'][1]=1

####################################################################################
# ...train the model...
####################################################################################

All_file_names
ense = 21
numbins = 15
allinds = np.arange(len(All_file_names))


avgvar_mA = np.zeros([len(All_file_names),numbins])
msebin_mA = np.zeros([len(All_file_names),numbins])
avgvar_sA = np.zeros([len(All_file_names),numbins])
msebin_sA = np.zeros([len(All_file_names),numbins])

mAl =np.array([])
sAl =np.array([])
yAl =np.array([])
gefsmAl = np.array([])
gefssAl = np.array([])

latoutAl = np.array([])
lonoutAl = np.array([])
tout = np.array([],dtype='datetime64[ns]')

for ddfa, bb in enumerate(range(0,len(All_file_names))):

    if bb == (len(All_file_names)-1):
        allinds = np.arange(len(All_file_names))
        test_fil_name = [All_file_names[allinds[0]]]
        val_fil_name = [All_file_names[allinds[-1]]]
        rest = np.delete(allinds,[bb,bb+1])
        train_fil_name = np.array(All_file_names)[rest].tolist()
    else:
        allinds = np.arange(len(All_file_names))
        test_fil_name = [All_file_names[allinds[bb+1]]]
        val_fil_name = [All_file_names[allinds[bb]]]
        rest = np.delete(allinds,[bb,bb+1])
        train_fil_name = np.array(All_file_names)[rest].tolist()
    
    print('#################################################')
    print('testing:',test_fil_name)
    print('validating:',val_fil_name)
    print('#################################################')
    
    num_samps_train = utilsProbSS_Ens.count_samps(train_fil_name,latind,lonind)
    num_samps_val = utilsProbSS_Ens.count_samps(val_fil_name,latind,lonind)

    print('...gathering data...')
    aa = utilsProbSS_Ens.deep_learning_generator_ss_mv(train_fil_name, num_samps_train*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x = adf[0]
    y = adf[1]
    
    ######## load testing ##############################################
    xpred = utilsProbSS_Ens.read_image_file(test_fil_name[0],latind,lonind)
    x_tst = np.reshape(xpred['predictor_matrix'],[xpred['predictor_matrix'].shape[0]*xpred['predictor_matrix'].shape[1],xpred['predictor_matrix'].shape[2]],order='F')
    y_tst = np.reshape(xpred['target_matrix'],[xpred['target_matrix'].shape[0]*xpred['target_matrix'].shape[1]],order='F')
    x_tst = utilsProbSS.normalize_images(x_tst,xpred['predictor_names'],norm_dict)[0]
    y_tst = utilsProbSS.normalize_images_targ(y_tst,xpred['target_name'],norm_dict_targ)[0]
    y_tst = np.expand_dims(y_tst,axis=1)
    
    
    #gather other variables of interest (time and lat/lon)
    op = xr.open_dataset(test_fil_name[0])
    lat = op['lat'][:].data
    lon = op['lon'][:].data
    lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],op['IVTm'][:].data.shape[0],1),order='F')
    lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],op['IVTm'][:].data.shape[0],1),order='F')
    time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 

    ######## load testing ##############################################
    
    #remove x and y where x ==0 
    rmind = np.where(utilsProbSS_Ens.denormalize_images_targ(x[:,0],'IVT_mean',norm_dict)==0)
    print('before shape:',y.shape)
    print('deleting:',rmind[0].shape,'indices')
    x = np.delete(x,rmind[0],0)
    y = np.delete(y,rmind[0],0)
    print('after shape:',y.shape)
    
    
    rmind = np.where(utilsProbSS_Ens.denormalize_images_targ(x_tst[:,0],'IVT_mean',norm_dict)==0)
    print('before shape test:',y_tst.shape)
    print('deleting test:',rmind[0].shape,'indices')
    x_tst = np.delete(x_tst,rmind[0],0)
    y_tst = np.delete(y_tst,rmind[0],0)
    print('after shape tst:',y_tst.shape)

    print('...Encoding Stations...')

    SUMID = np.unique(x[:,2]+x[:,3])
    #station ID integers 
    stID = np.zeros([x.shape[0],1])
    for jj,un in enumerate(SUMID):
        mats = np.where(x[:,2]+x[:,3]==un)
        stID[mats,:] = int(jj)
    stID=stID.astype(int)

    #station ID integers 
    stID_tst = np.zeros([x_tst.shape[0],1])
    for jj,un in enumerate(SUMID):
        mats = np.where(np.around(10000*np.round((x_tst[:,2]+x_tst[:,3]),4)).astype(int)==np.around(10000*np.round(un,4)).astype(int))[0]
        stID_tst[mats,:] = int(jj)
    stID_tst=stID_tst.astype(int)
    print('...done...')

    x=x[:,:2]
    x_tst=x_tst[:,:2]
    
    ### Model Build #### 
    
    in_shape = x.shape[1]
    print('In shape: ',in_shape)
    out_shape = 1
    print('Out shape: ',out_shape)
    
    max_id = np.max(stID)
    model = comsnngefs.build_emb_model(in_shape,2,[30,40],2,max_id,compile=True)
    model.summary()    
    valyr= val_fil_name[0].split('_GEFS')[0]
    valyr =valyr.split('_')[1]
    
    tstyr= test_fil_name[0].split('_GEFS')[0]
    tstyr =tstyr.split('_')[1]    
    newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/Gefs_ONLYTHREE_'+tstyr
    Wsave_name = newdir+'/cpf_CRPS_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    print("... loading....",Wsave_name)
    
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    model.load_weights(Wsave_name)
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    
    yhat = model([x_tst,stID_tst])
    
    m = yhat[:,0].numpy()
    m2 = yhat[:,0].numpy()
    s = np.abs(yhat[:,1].numpy())
    mm = yhat[:,0].numpy()
    
    WWRF = utilsProbSS_Ens.denormalize_images_targ(x_tst[:,0],'IVT_mean',norm_dict)
    WWRFs = utilsProbSS_Ens.denormalize_images_targ(x_tst[:,1],'IVT_std',norm_dict)
    y_tst = (utilsProbSS_Ens.denormalize_images_targ(y_tst,'IVTm',norm_dict_targ))
    
    #form ensembles 
    m_All = np.zeros(len(m))
    sss_All = np.zeros([len(m),ense])
    s_All = np.zeros(len(m))
    
    
    for ii in range(len(m)):
        sss = np.random.normal(m[ii],s[ii],ense)
        sln=np.std(sss)
        mln=np.mean(sss)
        [mln,sln]=comsnngefs.lognstat(m[ii],s[ii])
        s_All[ii]=s[ii]
        m_All[ii]=m[ii]
        sss_All[ii,:] =(sss) 

    m = m_All
    m[np.isnan(m)]=0
    s=s_All
    s[np.isnan(m)]=0
    
    y_tst = (y_tst)
    WWRF = (WWRF)
    distPP = s
    
    if ddfa == 0:
        sssAl = sss_All
    else: 
        sssAl = np.concatenate([sssAl,sss_All],axis=0)
    
    if bb == (len(All_file_names)-1):
        mAl = np.append(np.ndarray.flatten(m),mAl)
        yAl = np.append(np.ndarray.flatten(y_tst),yAl)
        sAl = np.append(np.ndarray.flatten(s),sAl)
        gefsmAl = np.append(np.ndarray.flatten(WWRF),gefsmAl)
        gefssAl = np.append(np.ndarray.flatten(WWRFs),gefssAl)
        latoutAl = np.append(np.ndarray.flatten(lat_out),latoutAl)
        lonoutAl = np.append(np.ndarray.flatten(lon_out),lonoutAl)
        tout = np.append(np.ndarray.flatten(time_out),tout)
    else:
        mAl = np.append(mAl,np.ndarray.flatten(m))
        yAl = np.append(yAl,np.ndarray.flatten(y_tst))
        sAl = np.append(sAl,np.ndarray.flatten(s))
        gefsmAl = np.append(gefsmAl,np.ndarray.flatten(WWRF))
        gefssAl = np.append(gefssAl,np.ndarray.flatten(WWRFs))
        latoutAl = np.append(latoutAl,np.ndarray.flatten(lat_out))
        lonoutAl = np.append(lonoutAl,np.ndarray.flatten(lon_out))
        tout = np.append(tout,np.ndarray.flatten(time_out))
    
    print('...binned spread skill...')
    [avgvar_m,msebin_m,avgvar_s,msebin_s]=comsnngefs.spreadskill(m,y_tst,distPP,numbins)
    
    avgvar_mA[bb,:] = np.squeeze(avgvar_m)
    msebin_mA[bb,:] = np.squeeze(msebin_m)
    avgvar_sA[bb,:] = np.squeeze(avgvar_s)
    msebin_sA[bb,:] = np.squeeze(msebin_s)
    
    print('RMSE PP:',comsnngefs.rmse(np.expand_dims(m,axis=1),y_tst))
    print('RMSE WWRF:',comsnngefs.rmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('CRMSE PP:',comsnngefs.crmse(np.expand_dims(m,axis=1),y_tst))
    print('CRMSE WWRF:',comsnngefs.crmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('BIAS PP:',comsnngefs.bias(np.expand_dims(m,axis=1),y_tst))
    print('BIAS WWRF:',comsnngefs.bias(np.expand_dims(WWRF,axis=1),y_tst))

    
newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/GEFS_out/'
if not os.path.exists(newdir):
        os.makedirs(newdir)

df_out = pd.DataFrame({'time':tout,'OBS':yAl,'Modelmean':gefsmAl,'Modelstd':gefssAl,'IVTmean':mAl,'IVTstd':sAl,'lat':latoutAl,'lon':lonoutAl})
df_out.to_pickle(newdir+'NN_CRPS_PP_gefs.pkl')

print('IVTm shape:', yAl.shape)
print('IVT mean shape:', mAl.shape)
print('IVT std shape:', sAl.shape)

np.savez(newdir+'NN_CRPS_PP_gefs.npz',lat=lat,lon=lon,IVTgefsm=mAl,IVTgefss=sAl,IVTm_gefs=yAl)
print('.done.')

print('#############################################################################')
print('############## *I hope that is okay baby* - gale beggy ######################')
print('#############################################################################')

print(df_out)
print(df_out.isnull().sum())
print(df_out.describe())

print('#############################################################################')
print('############## *I hope that is okay baby* - gale beggy ######################')
print('#############################################################################')