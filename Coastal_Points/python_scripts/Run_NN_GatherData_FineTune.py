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
from tensorflow.python.client import device_lib

import os

import utilsProb
import utilsProbSS
import comsnn

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

stepnum=int(sys.argv[1])
dirA = ['F'+f'{x:03}' for x in np.arange(0,126,6)]
print('#############################################')
print('post processing forecast:', dirA[stepnum-1])
print('#############################################')


#####################################################################################
#GPU cuda handling: 
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('#################################################')
    print('#################################################')
    print('WARNING: GPU device not found.')
    print('#################################################')
    print('#################################################')
else:
    print('#################################################')
    print('#################################################')
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
    print('#################################################')
    print('#################################################')
####################################################################################
print('We are here:',os.getcwd())
os.chdir('/glade/work/wchapman/AnEn/CNN/Coastal_Points_LogNormal/')


####################################################################################
# check region of interest folder
####################################################################################




dd = '/glade/scratch/wchapman/Reforecast/'
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
train_file_names = sorted([f for f in glob.glob(path + "/train/*_500mb_Clean.nc", recursive=True)])
for f in train_file_names:
    print(f)

print('Validating on')
validate_file_names = sorted([f for f in glob.glob(path + "/validate/*_500mb_Clean.nc", recursive=True)])
for f in validate_file_names:
    print(f)
    
print('Testing on')
test_file_names = sorted([f for f in glob.glob(path + "/test/*_500mb_Clean.nc", recursive=True)])
for f in test_file_names:
    print(f)
    
All_file_names =train_file_names + validate_file_names +test_file_names 

####################################################################################
#years to include in training. 
####################################################################################
# yearstoinclude = ['1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005',
#                   '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
yearstoinclude = ['2016','2017','2018']
res=[]
for jj in yearstoinclude:
    res.append([i for i in All_file_names if jj in i][0])
    
print(res)

####################################################################################
#region: 
####################################################################################

rang = 144
latlonfolder = '/glade/scratch/wchapman/AnEnCNN_good/Data/WestCoast/'
[latsDO,lonsDO,latind, lonind] = utilsProbSS.get_latlon_ind(latlonfolder)
latsDO.shape
latind = np.array(latind[:rang])
lonind = np.array(lonind[:rang])
latsDO = np.array(latsDO[:rang])
lonsDO = np.array(lonsDO[:rang])

norm_dict = utilsProbSS.get_image_normalization_params(res,latind,lonind)
sys.stdout.flush()
norm_dict_targ = utilsProbSS.get_image_normalization_params_targ(res,latind,lonind)

#no scaling the target
norm_dict_targ['IVTm'][0]=0
norm_dict_targ['IVTm'][1]=1

sys.stdout.flush()
num_samps_train = utilsProb.count_samps(train_file_names)
num_samps_val = utilsProb.count_samps(validate_file_names)
num_samps_test = utilsProb.count_samps(test_file_names)

print('number of training samples:',num_samps_train)
sys.stdout.flush()

print('number of validation samples:',num_samps_val)
sys.stdout.flush()
print('number of validation samples:',num_samps_test)
num_samps_tot = num_samps_train+num_samps_val+num_samps_test


####################################################################################
#INFERENCE! 
####################################################################################
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
sssAl = np.array([])
wwrfAl = np.array([])
latoutAl = np.array([])
lonoutAl = np.array([])
tout = np.array([],dtype='datetime64[ns]')

for ddfa, bb in enumerate(range(len(res)-3,len(res))):
    
    if bb == (len(res)-1):
        allinds = np.arange(len(res))
        test_fil_name = [res[allinds[-3]]]
        val_fil_name = [res[allinds[-1]]]
        rest = np.delete(allinds,[bb,bb+1])
        train_fil_name = np.array(res)[rest].tolist()
    else:
        allinds = np.arange(len(res))
        test_fil_name = [res[allinds[bb+1]]]
        val_fil_name = [res[allinds[bb]]]
        rest = np.delete(allinds,[bb,bb+1])
        train_fil_name = np.array(res)[rest].tolist()
        
    
    print('#################################################')
    print('testing:',test_fil_name)
    print('validating:',val_fil_name)
    print('#################################################')
    
    num_samps_train = utilsProb.count_samps(train_fil_name)
    num_samps_val = utilsProb.count_samps(val_fil_name)

    print('...gathering data...')
    aa = utilsProbSS.deep_learning_generator_ss_mv(train_fil_name, num_samps_train*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x = adf[0]
    y = adf[1]
    
    ######## load testing ##############################################
    xpred = utilsProbSS.read_image_file(test_fil_name[0],latind,lonind)
    x_tst = np.reshape(xpred['predictor_matrix'],[xpred['predictor_matrix'].shape[0]*xpred['predictor_matrix'].shape[1],xpred['predictor_matrix'].shape[2]],order='F')
    y_tst = np.reshape(xpred['target_matrix'],[xpred['target_matrix'].shape[0]*xpred['target_matrix'].shape[1]],order='F')
    x_tst = utilsProbSS.normalize_images(x_tst,xpred['predictor_names'],norm_dict)[0]
    y_tst = utilsProbSS.normalize_images_targ(y_tst,xpred['target_name'],norm_dict_targ)[0]
    y_tst = np.expand_dims(y_tst,axis=1)
    
    
    #gather other variables of interest (time and lat/lon)
    op = xr.open_dataset(test_fil_name[0])
    lat = op['lat'][:].data
    lon = op['lon'][:].data
    lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],xpred['predictor_matrix'].shape[0],1),order='F')
    lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],xpred['predictor_matrix'].shape[0],1),order='F')
    time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 
    ######## load testing ##############################################
    
    #remove x and y where x ==0 
    rmind = np.where(utilsProbSS.denormalize_images_targ(x[:,0],'IVT',norm_dict)==0)
    print('before shape:',y.shape)
    print('deleting:',rmind[0].shape,'indices')
    x = np.delete(x,rmind[0],0)
    y = np.delete(y,rmind[0],0)
    print('after shape:',y.shape)
    
    
    rmind = np.where(utilsProbSS.denormalize_images_targ(x_tst[:,0],'IVT',norm_dict)==0)
    print('before shape test:',y_tst.shape)
    print('deleting test:',rmind[0].shape,'indices')
    x_tst = np.delete(x_tst,rmind[0],0)
    y_tst = np.delete(y_tst,rmind[0],0)
    print('after shape tst:',y_tst.shape)

    print('...Encoding Stations...')

    SUMID = np.unique(x[:,6]+x[:,7])
    #station ID integers 
    stID = np.zeros([x.shape[0],1])
    for jj,un in enumerate(SUMID):
        mats = np.where(x[:,6]+x[:,7]==un)
        stID[mats,:] = int(jj)
    stID=stID.astype(int)

    #station ID integers 
    stID_tst = np.zeros([x_tst.shape[0],1])
    for jj,un in enumerate(SUMID):
        mats = np.where(np.around(10000*np.round((x_tst[:,6]+x_tst[:,7]),4)).astype(int)==np.around(10000*np.round(un,4)).astype(int))[0]
        stID_tst[mats,:] = int(jj)
    stID_tst=stID_tst.astype(int)
    print('...done...')
    
    x=x[:,:6]
    x_tst=x_tst[:,:6]
    
    ### Model Build #### 
    
    in_shape = x.shape[1]
    print('In shape: ',in_shape)
    out_shape = 1
    print('Out shape: ',out_shape)
    
    max_id = np.max(stID)
    model = comsnn.build_emb_model(in_shape,2,[30,40],2,max_id,compile=True)
    
    newout  = tf.keras.layers.Dense(32,activation='relu')(model.layers[-1].output)
    newout  = tf.keras.layers.Dense(2,activation='linear')(newout)
    model = tf.keras.models.Model(inputs = model.inputs, outputs = newout)
    opt = optimizer=tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='adam', loss=comsnn.crps_cost_function)    
    model.summary()
    
    valyr= val_fil_name[0].split('_500mb')[0]
    valyr =valyr.split('_')[2]
    
    tstyr= test_fil_name[0].split('_500mb')[0]
    tstyr =tstyr.split('_')[2]
    
    newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/ONLYTHREE_'+tstyr
    Wsave_name = newdir+'/cpf_CRPS_val_'+ valyr+'_test_FINETUNE_'+tstyr+'.ckpt'
    print("... loading....",Wsave_name)
    
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    model.load_weights(Wsave_name)
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    
    yhat = model([x_tst,stID_tst])
    
    m = yhat[:,0].numpy()
    m2 = yhat[:,0].numpy()
    s = np.abs(yhat[:,1].numpy())
    mm = yhat[:,0].numpy()

    WWRF = utilsProbSS.denormalize_images_targ(x_tst[:,0],'IVT',norm_dict)
    y_tst = (utilsProbSS.denormalize_images_targ(y_tst,'IVTm',norm_dict_targ))

    #form ensembles 
    m_All = np.zeros(len(m))
    sss_All = np.zeros([len(m),ense])
    s_All = np.zeros(len(m))
    
    
    for ii in range(len(m)):
        sss = np.random.normal(m[ii],s[ii],ense)
        sln=np.std(sss)
        mln=np.mean(sss)
        [mln,sln]=comsnn.lognstat(m[ii],s[ii])
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
    
    #this logic sucks... sorry. 
    if ddfa == 0:
        sssAl = sss_All
    elif bb == (len(res)-1):
        sssAl = np.concatenate([sss_All,sssAl],axis=0)
    else:
        sssAl = np.concatenate([sssAl,sss_All],axis=0)
    
    if bb == (len(res)-1):
        mAl = np.append(np.ndarray.flatten(m),mAl)
        yAl = np.append(np.ndarray.flatten(y_tst),yAl)
        sAl = np.append(np.ndarray.flatten(s),sAl)
        wwrfAl = np.append(np.ndarray.flatten(WWRF),wwrfAl)
        latoutAl = np.append(np.ndarray.flatten(lat_out),latoutAl)
        lonoutAl = np.append(np.ndarray.flatten(lon_out),lonoutAl)
        tout = np.append(np.ndarray.flatten(time_out),tout)
    else:
        mAl = np.append(mAl,np.ndarray.flatten(m))
        yAl = np.append(yAl,np.ndarray.flatten(y_tst))
        sAl = np.append(sAl,np.ndarray.flatten(s))
        wwrfAl = np.append(wwrfAl,np.ndarray.flatten(WWRF))
        latoutAl = np.append(latoutAl,np.ndarray.flatten(lat_out))
        lonoutAl = np.append(lonoutAl,np.ndarray.flatten(lon_out))
        tout = np.append(tout,np.ndarray.flatten(time_out))
    
    print('...binned spread skill...')
    [avgvar_m,msebin_m,avgvar_s,msebin_s]=comsnn.spreadskill(m,y_tst,distPP,numbins)
    
    avgvar_mA[bb,:] = np.squeeze(avgvar_m)
    msebin_mA[bb,:] = np.squeeze(msebin_m)
    avgvar_sA[bb,:] = np.squeeze(avgvar_s)
    msebin_sA[bb,:] = np.squeeze(msebin_s)
    
    print('RMSE PP:',comsnn.rmse(np.expand_dims(m,axis=1),y_tst))
    print('RMSE WWRF:',comsnn.rmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('CRMSE PP:',comsnn.crmse(np.expand_dims(m,axis=1),y_tst))
    print('CRMSE WWRF:',comsnn.crmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('BIAS PP:',comsnn.bias(np.expand_dims(m,axis=1),y_tst))
    print('BIAS WWRF:',comsnn.bias(np.expand_dims(WWRF,axis=1),y_tst))

newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/Reforecast_out/'
print('IVTm shape:', yAl.shape)
print('IVT mean shape:', mAl.shape)
print('IVT std shape:', sAl.shape)

#save out. 
df_out = pd.DataFrame({'time':tout,'OBS':yAl,'Model':wwrfAl,'IVTmean':mAl,'IVTstd':sAl,'lat':latoutAl,'lon':lonoutAl})
df_out.to_pickle(newdir+'NN_FineTune_CRPS_PP_ref2.pkl')
np.savez(newdir+'NN_FineTune_CRPS_PP_ref2.npz',lat=lat,lon=lon,IVTgefsm=mAl,IVTgefss=sAl,IVTm_gefs=yAl)

print('#############################################################################')
print('############## *I hope that is okay baby* - gale beggy ######################')
print('#############################################################################')
print('done')
print(df_out)
print(df_out.isnull().sum())
print(df_out.describe())
print('#############################################################################')
print('############## *I hope that is okay baby* - gale beggy ######################')
print('#############################################################################')