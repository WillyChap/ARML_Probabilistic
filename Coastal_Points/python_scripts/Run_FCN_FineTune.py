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
tf.random.set_seed(33) #patrick roy's number.
np.random.seed(33) #set numpy seed 
#epocs for taining and fine tuning
epos_ = 200
epos_2 = 200
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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
yearstoinclude = ['1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005',
                  '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
# yearstoinclude = ['2004','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
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
#Training! 
####################################################################################

allinds = np.arange(len(res))
for bb in range(len(res)-3,len(res)):
    
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
    print('validatiing:',val_fil_name)
    print('#################################################')
    
    num_samps_train = utilsProb.count_samps(train_fil_name)
    num_samps_val = utilsProb.count_samps(val_fil_name)

    print('...gathering data...')
    aa = utilsProbSS.deep_learning_generator_ss_mv(train_fil_name, num_samps_train*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x = adf[0]
    y = adf[1]
    aa = utilsProbSS.deep_learning_generator_ss_mv(val_fil_name, num_samps_val*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x_tst = adf[0]
    y_tst = adf[1]

    #remove x and y where x ==0 
    rmind = np.where(utilsProbSS.denormalize_images_targ(x[:,0],'IVT',norm_dict)==0)
    print('before shape:',y.shape)
    print('deleting:',rmind[0].shape,'values')
    x = np.delete(x,rmind[0],0)
    y = np.delete(y,rmind[0],0)
    print('after shape:',y.shape)
    
    rmind = np.where(utilsProbSS.denormalize_images_targ(x_tst[:,0],'IVT',norm_dict)==0)
    print('before shape test:',y_tst.shape)
    print('deleting test:',rmind[0].shape,'values')
    x_tst = np.delete(x_tst,rmind[0],0)
    y_tst = np.delete(y_tst,rmind[0],0)
    print('after shape tst:',y_tst.shape)   
    
    print('y_tst:',y_tst.shape[0])
    print('x_tst:',x_tst.shape[0])
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
        mats = np.where(x_tst[:,6]+x_tst[:,7]==un)
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
    
    model = comsnn.build_emb_model(in_shape,2,[],2,max_id,compile=True,lr=0.001)
    
    model.summary()
    valyr= val_fil_name[0].split('_500mb')[0]
    valyr =valyr.split('_')[2]
    tstyr= test_fil_name[0].split('_500mb')[0]
    tstyr =tstyr.split('_')[2]
    #save location:
    newdir = '/glade/scratch/wchapman/Reforecast/models/FCN_CRPS/' +fcast+'/ONLYTHREE_'+tstyr
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    Wsave_name = newdir+'/cpf_CRPS_FCN_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    
    print(Wsave_name)
    print(Wsave_name)

    modsave = tf.keras.callbacks.ModelCheckpoint(Wsave_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min',include_optimizer=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001,verbose=1)
    er_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto',baseline=None, restore_best_weights=False)
    net_in = tf.keras.layers.Input(shape=(in_shape))
    
    #train model
    histss = model.fit([x,stID], y,validation_data=([x_tst,stID_tst],y_tst), epochs=epos_,batch_size=100 ,verbose=2,callbacks=[modsave,reduce_lr,er_stop]);
    hist_df = pd.DataFrame(histss.history) 
   
    # or save to csv: 
    hist_csv_file = newdir+'/fithist_CRPS_'+ valyr+'_test_'+tstyr+'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


print('xnan:',np.sum(np.isnan(x)))
print('ynan:',np.sum(np.isnan(y)))

print('x_tstnan:',np.sum(np.isnan(x_tst)))
print('y_tstnan:',np.sum(np.isnan(y_tst)))
####################################################################################
#finally fine tuning it. 
####################################################################################

        
yearstoinclude = ['2016','2017','2018']
res=[]
for jj in yearstoinclude:
    res.append([i for i in All_file_names if jj in i][0])
    
print(res)

####################################################################################
#finally fine tuning it. 
####################################################################################


allinds = np.arange(len(res))

for bb in range(len(res)-3,len(res)):
    
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
    print('validatiing:',val_fil_name)
    print('#################################################')
    
    num_samps_train = utilsProb.count_samps(train_fil_name)
    num_samps_val = utilsProb.count_samps(val_fil_name)

    print('...gathering data...')
    aa = utilsProbSS.deep_learning_generator_ss_mv(train_fil_name, num_samps_train*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x = adf[0]
    y = adf[1]
    aa = utilsProbSS.deep_learning_generator_ss_mv(val_fil_name, num_samps_val*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x_tst = adf[0]
    y_tst = adf[1]

    #remove x and y where x ==0 
    rmind = np.where(utilsProbSS.denormalize_images_targ(x[:,0],'IVT',norm_dict)==0)
    print('before shape:',y.shape)
    print('deleting:',rmind[0].shape,'values')
    x = np.delete(x,rmind[0],0)
    y = np.delete(y,rmind[0],0)
    print('after shape:',y.shape)
    
    rmind = np.where(utilsProbSS.denormalize_images_targ(x_tst[:,0],'IVT',norm_dict)==0)
    print('before shape test:',y_tst.shape)
    print('deleting test:',rmind[0].shape,'values')
    x_tst = np.delete(x_tst,rmind[0],0)
    y_tst = np.delete(y_tst,rmind[0],0)
    print('after shape tst:',y_tst.shape)   
    
    print('y_tst:',y_tst.shape[0])
    print('x_tst:',x_tst.shape[0])
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
        mats = np.where(x_tst[:,6]+x_tst[:,7]==un)
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
    
    model = comsnn.build_emb_model(in_shape,2,[],2,max_id,compile=True,lr=0.0001)
        
    
    print('########### Loading old model weights name:',Wsave_name)
    print('########### Loading old model weights name:',Wsave_name)
    print('########### Loading old model weights name:',Wsave_name)
    print('########### Loading old model weights name:',Wsave_name)
    print('########### Loading old model weights name:',Wsave_name)
    
    newdir = '/glade/scratch/wchapman/Reforecast/models/FCN_CRPS/' +fcast+'/ONLYTHREE_'+tstyr
    Wsave_name = newdir+'/cpf_CRPS_FCN_val_'+ valyr+'_test_'+tstyr+'.ckpt'

    print('########### layers frozen ###########')
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    model.load_weights(Wsave_name).expect_partial()
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    
    #fine tuning layers:
#     for layer in model.layers[:]:
#         layer.trainable = False

#     # Check the trainable status of the individual layers

#     for layer in model.layers:
#         print(layer, layer.trainable)
        
        
#     newout  = tf.keras.layers.Dense(32,activation='relu')(model.layers[-1].output)
#     newout  = tf.keras.layers.Dense(2,activation='linear')(newout)
#     model = tf.keras.models.Model(inputs = model.inputs, outputs = newout)
#     opt = optimizer=tf.optimizers.Adam(learning_rate=0.0001)
#     model.compile(optimizer='adam', loss=comsnn.crps_cost_function)    
#     model.summary()
    
    
    
    
    
    valyr= val_fil_name[0].split('_500mb')[0]
    valyr =valyr.split('_')[2]
    tstyr= test_fil_name[0].split('_500mb')[0]
    tstyr =tstyr.split('_')[2]
    
    #save location:
    newdir = '/glade/scratch/wchapman/Reforecast/models/FCN_CRPS/' +fcast+'/ONLYTHREE_'+tstyr
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    Wsave_name = newdir+'/cpf_CRPS_FCN_val_'+ valyr+'_test_FINETUNE_'+tstyr+'.ckpt'
    
    print(Wsave_name)
    print(Wsave_name)

    modsave = tf.keras.callbacks.ModelCheckpoint(Wsave_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min',include_optimizer=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001,verbose=1)
    er_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto',baseline=None, restore_best_weights=False)
    net_in = tf.keras.layers.Input(shape=(in_shape))
    
    #train model
    histss = model.fit([x,stID], y,validation_data=([x_tst,stID_tst],y_tst), epochs=epos_2,batch_size=100 ,verbose=2,callbacks=[modsave,reduce_lr,er_stop]);
    hist_df = pd.DataFrame(histss.history) 
   
    # or save to csv: 
    hist_csv_file = newdir+'/fithist_FCN_FINETUNE_CRPS_'+ valyr+'_test_'+tstyr+'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        
print('done')