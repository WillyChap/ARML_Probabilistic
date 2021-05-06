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
tf.random.set_seed(33) #patrick roy's number.
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

####################################################################################
# ...train the model...
####################################################################################

norm_dict_targ['IVTm'][0]=0
norm_dict_targ['IVTm'][1]=1


allinds = np.arange(len(All_file_names))

for bb in range(0,len(All_file_names)):
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
    print('validatiing:',val_fil_name)
    print('#################################################')
    
    num_samps_train = utilsProbSS_Ens.count_samps(train_fil_name,latind,lonind)
    num_samps_val = utilsProbSS_Ens.count_samps(val_fil_name,latind,lonind)

    print('...gathering data...')
    aa = utilsProbSS_Ens.deep_learning_generator_ss_mv(train_fil_name, num_samps_train*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x = adf[0]
    y = adf[1]
    aa = utilsProbSS_Ens.deep_learning_generator_ss_mv(val_fil_name, num_samps_val*len(latind),norm_dict,norm_dict_targ,targ_LATinds=latind,targ_LONinds=lonind)
    adf = next(aa)
    x_tst = adf[0]
    y_tst = adf[1]

    #remove x and y where x == 0 
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
    
    
    print('y_tst:',y_tst.shape[0])
    print('x_tst:',x_tst.shape[0])
    
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
        mats = np.where(x_tst[:,2]+x_tst[:,3]==un)
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
    print(tstyr)
    #save location:
    newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/Gefs_ONLYTHREE_'+tstyr
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    Wsave_name = newdir+'/cpf_CRPS_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    print(Wsave_name)
    
    print(Wsave_name)
    print(Wsave_name)
    
    modsave = tf.keras.callbacks.ModelCheckpoint(Wsave_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min',include_optimizer=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001,verbose=1)
    er_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto',baseline=None, restore_best_weights=False)
    net_in = tf.keras.layers.Input(shape=(in_shape))
    
    #train model
    histss = model.fit([x,stID], y,validation_data=([x_tst,stID_tst],y_tst), epochs=200,batch_size=50 ,verbose=True,callbacks=[modsave,reduce_lr,er_stop]);
    
    hist_df = pd.DataFrame(histss.history) 

   
    # or save to csv: 
    hist_csv_file = newdir+'/fithist_CRPS_'+ valyr+'_test_'+tstyr+'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
