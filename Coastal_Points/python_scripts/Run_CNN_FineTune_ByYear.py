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


import Unet_b
import utils_CNN
import coms
from tensorflow.python.client import device_lib
####################################################################################
tf.random.set_seed(33) #patrick roy's number.
####################################################################################

stepnum=int(sys.argv[1])
years_back = int(sys.argv[2])
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

yearstoinclude = pd.date_range(str(int(2018)-years_back), periods=years_back+1, freq="Y").year.astype('str').tolist()
print(yearstoinclude)
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
    aa = utils_CNN.deep_learning_generator(train_fil_name, num_samps_train,norm_dict,norm_dict_targ)
    adf = next(aa)
    x = adf[0]
    y = adf[1]

    aa = utils_CNN.deep_learning_generator(val_fil_name, num_samps_val,norm_dict,norm_dict_targ)
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
    newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/StartingYear'+str(2018-years_back)+'/CNN_'+tstyr
    print(newdir)
    print(newdir)
    print(newdir)

    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    Wsave_name = newdir+'/cpf_CRPS_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    
    print('....saving....',Wsave_name)
    print('....saving....',Wsave_name)

    modsave = tf.keras.callbacks.ModelCheckpoint(Wsave_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min',include_optimizer=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001,verbose=1)
    er_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto',baseline=None, restore_best_weights=False)
    
    #train model
    histss = model.fit(x, y,validation_data=(x_tst,y_tst), epochs=70,batch_size=50 ,verbose=True,callbacks=[modsave,reduce_lr,er_stop]);
    hist_df = pd.DataFrame(histss.history) 
   
    # or save to csv: 
    hist_csv_file = newdir+'/fithist_CRPS_'+ valyr+'_test_'+tstyr+'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


####################################################################################
#Check for any NAN values
####################################################################################
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

All_file_names

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
    aa = utils_CNN.deep_learning_generator(train_fil_name, num_samps_train,norm_dict,norm_dict_targ)
    adf = next(aa)
    x = adf[0]
    y = adf[1]

    aa = utils_CNN.deep_learning_generator(val_fil_name, num_samps_val,norm_dict,norm_dict_targ)
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
    newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/StartingYear'+str(2018-years_back)+'/CNN_'+tstyr
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    print(newdir)
    print(newdir)
    print(newdir)
    print(newdir)
    print(newdir)
    print(newdir)
    print(newdir)
    print(newdir)
    print(newdir)
    Wsave_name = newdir+'/cpf_CRPS_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    
    print('########### Loading old model weights name:',Wsave_name)
    print('########### Loading old model weights name:',Wsave_name)
    print('########### layers frozen ###########')
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    model.load_weights(Wsave_name)
    ### CHANGE THIS FOR PRODUCTION RUNS ###########
    for layer in model.layers[:]:
        layer.trainable = False

    # Check the trainable status of the individual layers

    for layer in model.layers:
        print(layer, layer.trainable)


    newout  = tf.keras.layers.Conv2D(32, 3, activation='relu',padding='same')(model.layers[-1].output)
    newout  = tf.keras.layers.Conv2D(2, 3, activation='linear',padding='same')(newout)
    model = tf.keras.models.Model(inputs = model.inputs, outputs = newout)
    opt = optimizer=tf.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss=coms.crps_cost_function)
    model.summary()

    for layer in model.layers:
        print(layer, layer.trainable)
        
    
    Wsave_name = newdir+'/cpf_CRPS_FINETUNE_val_'+ valyr+'_test_'+tstyr+'.ckpt'
    
    print('########### Saving new model name:',Wsave_name)
    modsave = tf.keras.callbacks.ModelCheckpoint(Wsave_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min',include_optimizer=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001,verbose=1)
    er_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto',baseline=None, restore_best_weights=False)
    
    #train model
    histss = model.fit(x, y,validation_data=(x_tst,y_tst), epochs=200,batch_size=20 ,verbose=True,callbacks=[modsave,reduce_lr,er_stop]);
    hist_df = pd.DataFrame(histss.history) 
   
    # or save to csv: 
    hist_csv_file = newdir+'/fithist_CRPS_FINETUNE_'+ valyr+'_test_'+tstyr+'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


        
print('done')