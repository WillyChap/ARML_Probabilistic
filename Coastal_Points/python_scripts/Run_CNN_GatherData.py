####################################################################################
### Import Packages ### run in tfp environment: 
####################################################################################
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

import numpy.matlib
import Unet_b
import utils_CNN
import coms
from tensorflow.python.client import device_lib
####################################################################################
####################################################################################

stepnum=int(sys.argv[1])
dirA = ['F'+f'{x:03}' for x in np.arange(0,126,6)]
print('#############################################')
print('post processing forecast:', dirA[stepnum-1])
print('#############################################')

####################################################################################
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

latlonfolder = '/glade/scratch/wchapman/AnEnCNN_good/Data/WestCoast/'
lat = np.arange(22,60.5,.5)
lon = np.arange(200.0,249.375+.625,.625)
MapPoints = np.zeros([lat.shape[0],lon.shape[0]])
print('...Searching...:',latlonfolder)
path = latlonfolder
pp_file_names = [f for f in glob.glob(path + "/lat*lon*", recursive=True)]
pp_file_names = sorted(pp_file_names)
####################################################################################
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
####################################################################################


yearstoinclude = ['1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005',
                  '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
res=[]
for jj in yearstoinclude:
    res.append([i for i in All_file_names if jj in i][0])
    
print('trainging yearss',res)

####################################################################################
# normalization dictionaries
####################################################################################
norm_dict = utils_CNN.get_image_normalization_params(res)
sys.stdout.flush()
norm_dict_targ = utils_CNN.get_image_normalization_params_targ(res)
norm_dict_targ['IVTm'][0]=0
norm_dict_targ['IVTm'][1]=1
####################################################################################
#finally training
####################################################################################


All_file_names
allinds = np.arange(len(res))

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
wwrfAl = np.array([])
latoutAl = np.array([])
lonoutAl = np.array([])
tout = np.array([],dtype='datetime64[ns]')

print('######### ASSEMBLE!!!! ###########')
#find lat/lons of interest:
rang = 144
latlonfolder = '/glade/scratch/wchapman/AnEnCNN_good/Data/WestCoast/'
[latsDO,lonsDO,latind, lonind] = utilsProbSS.get_latlon_ind(latlonfolder)
latsDO.shape
latind = np.array(latind[:rang])
lonind = np.array(lonind[:rang])
latsDO = np.array(latsDO[:rang])
lonsDO = np.array(lonsDO[:rang])
print('########### DONE!!!! #############')



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
    print('validatiing:',val_fil_name)
    print('#################################################')
    
    num_samps_train = utilsProb.count_samps(train_fil_name)
    num_samps_val = utilsProb.count_samps(val_fil_name)

    print('...gathering data...')
    aa = utils_CNN.deep_learning_generator(train_fil_name, num_samps_train,norm_dict,norm_dict_targ)
    adf = next(aa)
    x = adf[0]
    y = adf[1]

    aa = utils_CNN.deep_learning_generator_nordnm(test_fil_name, num_samps_val,norm_dict,norm_dict_targ)
    adf = next(aa)
    x_tst = adf[0]
    y_tst = adf[1]
    
    
#     ### Model Build #### 
    img_height = x.shape[1]
    img_width = x.shape[2]
    num_channels = x.shape[3]
    img_shape = (img_height, img_width, num_channels)
    nummy_classes = 2

    model = Unet_b.model_simple_unet_initializer(img_shape=img_shape, cost_func=coms.crps_cost_function,num_classes=nummy_classes,num_levels = 1,num_layers =2, num_bottleneck = 3, filter_size_start =16, batch_norm=None, kernel_size = 3, 
                                     bottleneck_dilation = True, bottleneck_sum_activation =False)

    valyr= val_fil_name[0].split('_500mb')[0]
    valyr =valyr.split('_')[2]
    tstyr= test_fil_name[0].split('_500mb')[0]
    tstyr =tstyr.split('_')[2]
#     #save location:
    newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/CNN_'+tstyr
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    Wsave_name = newdir+'/cpf_CRPS_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    
    print('########### model name:',Wsave_name)
    print('########### model name:',Wsave_name)
    
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    model.load_weights(Wsave_name)
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    
    yhat = model(x_tst)
    yhat = yhat.numpy()
    
    m = np.ndarray.flatten(yhat[:,latind,lonind,0],order='F')
    m2 = np.ndarray.flatten(yhat[:,latind,lonind,0],order='F')
    mm = np.ndarray.flatten(yhat[:,latind,lonind,0],order='F')
    s = np.abs(np.ndarray.flatten(yhat[:,latind,lonind,1],order='F'))
    
    WWRF = utilsProbSS.denormalize_images_targ((x_tst[:,:,:,0]),'IVT',norm_dict)
    WWRF = np.ndarray.flatten(WWRF[:,latind,lonind],order='F')
    
    y_tst = (utilsProbSS.denormalize_images_targ(np.ndarray.flatten(y_tst[:,latind,lonind,0],order='F'),'IVTm',norm_dict_targ))
    y_tst=np.expand_dims(y_tst,axis=1)
    
    #gather other variables of interest (time and lat/lon)
    op = xr.open_dataset(test_fil_name[0])
    lat = op['lat'][:].data
    lon = op['lon'][:].data
    lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],x_tst.shape[0],1),order='F')
    lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],x_tst.shape[0],1),order='F')
    time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 
    

    #form ensembles 
    m_All = np.zeros(len(m))
    sss_All = np.zeros([len(m),ense])
    s_All = np.zeros(len(m))
    
    
    for ii in range(len(m)):
        sss = np.random.normal(m[ii],s[ii],ense)
        sln=np.std(sss)
        mln=np.mean(sss)
        [mln,sln]=coms.lognstat(m[ii],s[ii])
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
    [avgvar_m,msebin_m,avgvar_s,msebin_s]=coms.spreadskill(m,y_tst,distPP,numbins)
    
    avgvar_mA[bb,:] = np.squeeze(avgvar_m)
    msebin_mA[bb,:] = np.squeeze(msebin_m)
    avgvar_sA[bb,:] = np.squeeze(avgvar_s)
    msebin_sA[bb,:] = np.squeeze(msebin_s)
    
    print('RMSE PP:',coms.rmse(np.expand_dims(m,axis=1),y_tst))
    print('RMSE WWRF:',coms.rmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('CRMSE PP:',coms.crmse(np.expand_dims(m,axis=1),y_tst))
    print('CRMSE WWRF:',coms.crmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('BIAS PP:',coms.bias(np.expand_dims(m,axis=1),y_tst))
    print('BIAS WWRF:',coms.bias(np.expand_dims(WWRF,axis=1),y_tst))
    
newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/Reforecast_out/'
print('IVTm shape:', yAl.shape)
print('IVT mean shape:', mAl.shape)
print('IVT std shape:', sAl.shape)

df_out = pd.DataFrame({'time':tout,'OBS':yAl,'Model':wwrfAl,'IVTmean':mAl,'IVTstd':sAl,'lat':latoutAl,'lon':lonoutAl})






df_out.to_pickle(newdir+'NN_CRPS_CNNPP_ref2.pkl')
np.savez(newdir+'NN_CRPS_CNNPP_ref2.npz',lat=lat,lon=lon,IVTgefsm=mAl,IVTgefss=sAl,IVTm_gefs=yAl)

df233= df_out
print(df_out)
print(df_out.isnull().sum())
print(df_out.describe())
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

All_file_names

All_file_names

allinds = np.arange(len(res))

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
wwrfAl = np.array([])
latoutAl = np.array([])
lonoutAl = np.array([])
tout = np.array([],dtype='datetime64[ns]')

print('######### ASSEMBLE!!!! ###########')
#find lat/lons of interest:
rang = 144
latlonfolder = '/glade/scratch/wchapman/AnEnCNN_good/Data/WestCoast/'
[latsDO,lonsDO,latind, lonind] = utilsProbSS.get_latlon_ind(latlonfolder)
latsDO.shape
latind = np.array(latind[:rang])
lonind = np.array(lonind[:rang])
latsDO = np.array(latsDO[:rang])
lonsDO = np.array(lonsDO[:rang])
print('########### DONE!!!! #############')



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
    print('validatiing:',val_fil_name)
    print('#################################################')
    
    num_samps_train = utilsProb.count_samps(train_fil_name)
    num_samps_val = utilsProb.count_samps(val_fil_name)

    print('...gathering data...')
    aa = utils_CNN.deep_learning_generator(train_fil_name, num_samps_train,norm_dict,norm_dict_targ)
    adf = next(aa)
    x = adf[0]
    y = adf[1]

    aa = utils_CNN.deep_learning_generator_nordnm(test_fil_name, num_samps_val,norm_dict,norm_dict_targ)
    adf = next(aa)
    x_tst = adf[0]
    y_tst = adf[1]
    
    
#     ### Model Build #### 
    img_height = x.shape[1]
    img_width = x.shape[2]
    num_channels = x.shape[3]
    img_shape = (img_height, img_width, num_channels)
    nummy_classes = 2

    model = Unet_b.model_simple_unet_initializer(img_shape=img_shape, cost_func=coms.crps_cost_function,num_classes=nummy_classes,num_levels = 1,num_layers =2, num_bottleneck = 3, filter_size_start =16, batch_norm=None, kernel_size = 3, 
                                     bottleneck_dilation = True, bottleneck_sum_activation =False)

    newout  = tf.keras.layers.Conv2D(32, 3, activation='relu',padding='same')(model.layers[-1].output)
    newout  = tf.keras.layers.Conv2D(2, 3, activation='linear',padding='same')(newout)
    model = tf.keras.models.Model(inputs = model.inputs, outputs = newout)
    opt = optimizer=tf.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss=coms.crps_cost_function)
    model.summary()
    
    valyr= val_fil_name[0].split('_500mb')[0]
    valyr =valyr.split('_')[2]
    tstyr= test_fil_name[0].split('_500mb')[0]
    tstyr =tstyr.split('_')[2]
#     #save location:
    
    newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/CNN_'+tstyr
    Wsave_name = newdir+'/cpf_CRPS_FINETUNE_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    print('########### Loading old model weights name:',Wsave_name)
    model.load_weights(Wsave_name)
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    
    
    yhat = model(x_tst)
    yhat = yhat.numpy()
    
    m = np.ndarray.flatten(yhat[:,latind,lonind,0],order='F')
    m2 = np.ndarray.flatten(yhat[:,latind,lonind,0],order='F')
    mm = np.ndarray.flatten(yhat[:,latind,lonind,0],order='F')
    s = np.abs(np.ndarray.flatten(yhat[:,latind,lonind,1],order='F'))
    
    WWRF = utilsProbSS.denormalize_images_targ((x_tst[:,:,:,0]),'IVT',norm_dict)
    WWRF = np.ndarray.flatten(WWRF[:,latind,lonind],order='F')
    
    y_tst = (utilsProbSS.denormalize_images_targ(np.ndarray.flatten(y_tst[:,latind,lonind,0],order='F'),'IVTm',norm_dict_targ))
    y_tst=np.expand_dims(y_tst,axis=1)
    
    #gather other variables of interest (time and lat/lon)
    op = xr.open_dataset(test_fil_name[0])
    lat = op['lat'][:].data
    lon = op['lon'][:].data
    lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],x_tst.shape[0],1),order='F')
    lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],x_tst.shape[0],1),order='F')
    time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 
    

    #form ensembles 
    m_All = np.zeros(len(m))
    sss_All = np.zeros([len(m),ense])
    s_All = np.zeros(len(m))
    
    
    for ii in range(len(m)):
        sss = np.random.normal(m[ii],s[ii],ense)
        sln=np.std(sss)
        mln=np.mean(sss)
        [mln,sln]=coms.lognstat(m[ii],s[ii])
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
    [avgvar_m,msebin_m,avgvar_s,msebin_s]=coms.spreadskill(m,y_tst,distPP,numbins)
    
    avgvar_mA[bb,:] = np.squeeze(avgvar_m)
    msebin_mA[bb,:] = np.squeeze(msebin_m)
    avgvar_sA[bb,:] = np.squeeze(avgvar_s)
    msebin_sA[bb,:] = np.squeeze(msebin_s)
    
    print('RMSE PP:',coms.rmse(np.expand_dims(m,axis=1),y_tst))
    print('RMSE WWRF:',coms.rmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('CRMSE PP:',coms.crmse(np.expand_dims(m,axis=1),y_tst))
    print('CRMSE WWRF:',coms.crmse(np.expand_dims(WWRF,axis=1),y_tst))
    print('BIAS PP:',coms.bias(np.expand_dims(m,axis=1),y_tst))
    print('BIAS WWRF:',coms.bias(np.expand_dims(WWRF,axis=1),y_tst))
    
newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/Reforecast_out/'
print('IVTm shape:', yAl.shape)
print('IVT mean shape:', mAl.shape)
print('IVT std shape:', sAl.shape)

df_out = pd.DataFrame({'time':tout,'OBS':yAl,'Model':wwrfAl,'IVTmean':mAl,'IVTstd':sAl,'lat':latoutAl,'lon':lonoutAl})
df_out.to_pickle(newdir+'NN_CRPS_CNNPP_FINETUNEref2.pkl')
np.savez(newdir+'NN_CRPS_CNNPP_FINETUNEref2.npz',lat=lat,lon=lon,IVTgefsm=mAl,IVTgefss=sAl,IVTm_gefs=yAl)
        
print('done')
print(df_out)
print(df_out.isnull().sum())
print(df_out.describe())


print(df233)
print(df233.isnull().sum())
print(df233.describe())