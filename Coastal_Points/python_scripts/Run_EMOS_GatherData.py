## ### Import Packages ### run in tfp environment: 
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

import Unet_b
import utils_CNN
import coms
import comsnn
from tensorflow.python.client import device_lib
####################################################################################
tf.random.set_seed(33) #patrick roy's number.
####################################################################################
sys.stdout.flush()
# Params
stepnum=int(sys.argv[1])
# years_back = int(sys.argv[2])

# stepnum=6
years_back = 33

dirA = ['F'+f'{x:03}' for x in np.arange(0,126,6)]
print('#############################################')
print('gathering forecast:', dirA[stepnum-1])
print('#############################################')

sys.stdout.flush()
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

sys.stdout.flush()
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
####################################################################################
####################################################################################
sys.stdout.flush()

dd = '/glade/scratch/wchapman/Reforecast/'
os.chdir(dd)
sys.stdout.flush()
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

batch_num = 250
epochs = 100
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
sys.stdout.flush()

####################################################################################
####################################################################################
yearstoinclude = pd.date_range(str(int(2018)-years_back), periods=years_back+1, freq="Y").year.astype('str').tolist()
print(yearstoinclude)
res=[]
for jj in yearstoinclude:
    res.append([i for i in All_file_names if jj in i][0])
    
print('trainging yearss',res)
sys.stdout.flush()
####################################################################################
# normalization dictionaries
####################################################################################
norm_dict = utils_CNN.get_image_normalization_params(res)
sys.stdout.flush()
norm_dict_targ = utils_CNN.get_image_normalization_params_targ(res)
norm_dict_targ['IVTm'][0]=0
norm_dict_targ['IVTm'][1]=1

norm_dict['IVT'][0]=0
norm_dict['IVT'][1]=1
####################################################################################
#finally training
####################################################################################
All_file_names
allinds = np.arange(len(res))

mAl =np.array([])
sAl =np.array([])
yAl =np.array([])
wwrfAl = np.array([])
latoutAl = np.array([])
lonoutAl = np.array([])
tout = np.array([],dtype='datetime64[ns]')
sys.stdout.flush()
sys.stdout.flush()

for bb in range(len(res)-3,len(res)):
    print('count',bb)
    
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
#     print('validatiing:',val_fil_name)
    print('#################################################')
    
#     num_samps_train = utilsProb.count_samps(train_fil_name)
    num_samps_test = utilsProb.count_samps(test_fil_name)

    print('...Load Data...')
    aa = utils_CNN.deep_learning_generator_nordnm(test_fil_name, num_samps_test,norm_dict,norm_dict_targ)
    adf = next(aa)
    x_tst = adf[0]
    y_tst = adf[1]
    
    #open time and lat/lon variables
    op = xr.open_dataset(test_fil_name[0])
    lat = op['lat'][:].data
    lon = op['lon'][:].data
    time_out = op['time'][:].data
    lat_out = np.matlib.repmat(lat[latind],x_tst.shape[0],1)
    lon_out = np.matlib.repmat(lon[lonind],x_tst.shape[0],1)
    time_out
    
    #yearstring:
    valyr= val_fil_name[0].split('_500mb')[0]
    valyr =valyr.split('_')[2]
    tstyr= test_fil_name[0].split('_500mb')[0]
    tstyr =tstyr.split('_')[2]
    
    #gather other variables of interest (time and lat/lon)
    op = xr.open_dataset(test_fil_name[0])
    lat = op['lat'][:].data
    lon = op['lon'][:].data
    lat_out = np.matlib.repmat(lat[latind],x_tst.shape[0],1)
    lon_out = np.matlib.repmat(lon[lonind],x_tst.shape[0],1)
    time_out = op['time'][:].data
    
    
    
    mAlt =np.array([])
    sAlt =np.array([])
    yAlt =np.array([])
    wwrfAlt = np.array([])
    latoutAlt = np.array([])
    lonoutAlt = np.array([])
    toutt = np.array([],dtype='datetime64[ns]')
    
    sys.stdout.flush()
    #make new  model for each lat/lon location
    for hh,lala in enumerate(latind):
        print("number train:",hh)
    
        x_sp = x_tst[:,lala,lonind[hh],:].squeeze()
#         x_sp = np.expand_dims(x[:,lala,lonind[hh],0],axis=1)
        y_sp = y_tst[:,lala,lonind[hh]].squeeze()
        
        in_shape=x_sp.shape[1]
        model = comsnn.build_model(in_shape,2,[],compile=True,lr=0.01)
        
         #     #save location:#     ##     ##     ##     #
        newdir = '/glade/work/wchapman/AnEn/CNN/Coastal_Points_LogNormal/perm_models/' +fcast+'/EMOS_'+tstyr+ '/lat'+str(latsDO[hh])+'_lon'+str(lonsDO[hh]+360)
        Wsave_name = newdir+'/EMOS_FineTune_onlylast_CRPS_val_'+ valyr+'_test_'+tstyr+'_lat'+str(latsDO[hh])+'_lon'+str(lonsDO[hh]+360)+'.ckpt'
        ### model load weights ###########
        print('Loading:',Wsave_name)
        model.load_weights(Wsave_name).expect_partial()
        ### model load weights ###########
        yhat = np.abs(model.predict(x_sp))
        m = yhat[:,0]
        s = yhat[:,1]
        
        
        
        mAlt = np.append(mAlt,m)
        sAlt = np.append(sAlt,s)
        yAlt = np.append(yAlt,y_sp)
        latoutAlt = np.append(latoutAlt,lat_out[:,hh])
        lonoutAlt = np.append(lonoutAlt,lon_out[:,hh])
        toutt = np.append(toutt,time_out)
        wwrfAlt = np.append(wwrfAlt,x_sp[:,0].squeeze())
        
#     create_way to append here:

    if bb == (len(res)-1):
        mAl = np.append(mAlt,mAl)
        sAl = np.append(sAlt,sAl)
        yAl = np.append(yAlt,yAl)
        latoutAl = np.append(latoutAlt,latoutAl)
        lonoutAl = np.append(lonoutAlt,lonoutAl)
        tout = np.append(toutt,tout)
        wwrfAl = np.append(wwrfAlt,wwrfAl)
    else:
        mAl = np.append(mAl,mAlt)
        sAl = np.append(sAl,sAlt)
        yAl = np.append(yAl,yAlt)
        latoutAl = np.append(latoutAl,latoutAlt)
        lonoutAl = np.append(lonoutAl,lonoutAlt)
        tout = np.append(tout,toutt)
        wwrfAl = np.append(wwrfAl,wwrfAlt)
    
newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/'+fcast+'/Reforecast_out/'
print('IVTm shape:', yAl.shape)
print('IVT mean shape:', mAl.shape)
print('IVT std shape:', sAl.shape)

df_out = pd.DataFrame({'time':tout,'OBS':yAl,'Model':wwrfAl,'IVTmean':mAl,'IVTstd':sAl,'lat':latoutAl,'lon':lonoutAl})
df_out.to_pickle(newdir+'EMOS_FINETUNE_CRPS_ref2.pkl')
        
print('done')
print(df_out)
print(df_out.isnull().sum())
print(df_out.describe())

sys.stdout.flush()
sys.stdout.flush()
