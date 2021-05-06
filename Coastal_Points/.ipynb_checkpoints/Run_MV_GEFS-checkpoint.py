#note! still need to clean all of the GEFS files. 
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

#mapping
import xarray as xr
####################################################################################
####################################################################################
####################################################################################

####################################################################################

fcast=sys.argv[1]  #example input 'F048'

####################################################################################

print('IN THIS DIRECTORY:',"/glade/scratch/wchapman/GEFS/"+fcast+"/*"+fcast+"_Clean.nc")

gefs_file_names = sorted([f for f in glob.glob("/glade/scratch/wchapman/GEFS/"+fcast+"/*"+fcast+"_Clean.nc", recursive=True)])
for f in gefs_file_names:
    print(f)
    

# GEFS_raw:
latlonfolder = '/glade/scratch/wchapman/AnEnCNN_good/Data/WestCoast/'
[latsDO,lonsDO,latind,lonind] = utilsProbSS.get_latlon_ind_gefs(latlonfolder)
rang = len(latsDO)
print('...# number of lat/lon: ',rang)
latind = np.array(latind[:rang])
lonind = np.array(lonind[:rang])
latsDO = np.array(latsDO[:rang])
lonsDO = np.array(lonsDO[:rang])
latsDO.shape


#Mean variance correction: 
print('We are here:',os.getcwd())
os.chdir('/glade/work/wchapman/AnEn/CNN/Coastal_Points_LogNormal/')
#reloading utils after changes:
importlib.reload(utilsProbSS)
importlib.reload(utilsProb)


    
latlonfolder = '/glade/scratch/wchapman/AnEnCNN_good/Data/WestCoast/'
[latsDO,lonsDO,latind,lonind] = utilsProbSS.get_latlon_ind_gefs(latlonfolder)
latind = np.array(latind[:rang])
lonind = np.array(lonind[:rang])
latsDO = np.array(latsDO[:rang])
lonsDO = np.array(lonsDO[:rang])
latsDO.shape

latoutAl = np.array([])
lonoutAl = np.array([])
tout = np.array([],dtype='datetime64[ns]')


for bigcount, bb in enumerate(range(0,len(gefs_file_names))):

    allinds = np.arange(len(gefs_file_names))
    val_fil_name = [gefs_file_names[allinds[bb]]]
    rest = np.delete(allinds,[bb])
    train_fil_name = np.array(gefs_file_names)[rest].tolist()
        
    print('#################################################')
    print('training:',train_fil_name)
    print('validating:',val_fil_name)
    print('#################################################')
    
    for aa,f in enumerate(train_fil_name):
        ff = Dataset(f)
    
        if aa==0:
            IVTgefs = ff['IVT'][:].data
            IVTm_gefs = ff['IVTm'][:].data
            latgefs = ff['lat'][:]
            longefs = ff['lon'][:]
        else: 
            IVTgefs = np.concatenate([IVTgefs, ff['IVT'][:].data],axis=0)
            IVTm_gefs = np.concatenate([IVTm_gefs, ff['IVTm'][:].data],axis=0)
        break
            
    IVTgefsm = np.mean(IVTgefs,axis=1);
    IVTgefsv = np.var(IVTgefs,axis=1);
    IVTgefsv = np.mean(IVTgefsv[:,latind,lonind],axis=0) #<s^2> 
    
    print('Training IVT shape:',IVTm_gefs.shape)
    print('...Gathering Training...')
    MeanErrorGEFS = np.zeros(len(latind))
    ErrVarEnseMeanTrainingGEFS = np.zeros(len(latind))
    
    for rr in range(len(latind)):
        MeanErrorGEFS[rr] = np.mean(IVTgefsm[:,latind[rr],lonind[rr]]- IVTm_gefs[:,latind[rr],lonind[rr]],axis=0)
        ErrVarEnseMeanTrainingGEFS[rr] =np.var(IVTgefsm[:,latind[rr],lonind[rr]] - IVTm_gefs[:,latind[rr],lonind[rr]])
    
    datf=Dataset(val_fil_name[0])
    IVTgefs_pp = datf['IVT'][:].data
    IVTmgefs_pp = datf['IVTm'][:].data
    
    IVTgefs_ppm = np.mean(IVTgefs_pp,axis=1)
    IVTgefs_ppv = np.var(IVTgefs_pp,axis=1)
    
    for rr in range(len(latind)):
        IVTgefs_ppm[:,latind[rr],lonind[rr]] = IVTgefs_ppm[:,latind[rr],lonind[rr]]-MeanErrorGEFS[rr];                             #debiased forecast.
        IVTgefs_ppv[:,latind[rr],lonind[rr]] = (IVTgefs_ppv[:,latind[rr],lonind[rr]]/IVTgefsv[rr])*ErrVarEnseMeanTrainingGEFS[rr]  #variance corrected forecast. 
    
    
    if bigcount == 0:
        IVT_ppm_gefs = np.ndarray.flatten(IVTgefs_ppm[:,latind,lonind],order='F')
        IVT_ppv_gefs = np.ndarray.flatten(IVTgefs_ppv[:,latind,lonind],order='F')
        IVTm_pp_gefs = np.ndarray.flatten(IVTmgefs_pp[:,latind,lonind],order='F')
        op = xr.open_dataset(val_fil_name[0])
        
        lat = op['lat'][:].data
        lon = op['lon'][:].data
        lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],op['time'].shape[0],1),order='F')
        lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],op['time'].shape[0],1),order='F')
        time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 
        
        latoutAl = np.append(latoutAl,np.ndarray.flatten(lat_out))
        lonoutAl = np.append(lonoutAl,np.ndarray.flatten(lon_out))
        tout = np.append(tout,np.ndarray.flatten(time_out))
        
        
    else: 
        IVT_ppm_gefs = np.concatenate([IVT_ppm_gefs,np.ndarray.flatten(IVTgefs_ppm[:,latind,lonind],order='F')],axis=0)
        IVT_ppv_gefs = np.concatenate([IVT_ppv_gefs,np.ndarray.flatten(IVTgefs_ppv[:,latind,lonind],order='F')],axis=0)
        IVTm_pp_gefs = np.concatenate([IVTm_pp_gefs,np.ndarray.flatten(IVTmgefs_pp[:,latind,lonind],order='F')],axis=0)
        
        op = xr.open_dataset(val_fil_name[0])
        lat = op['lat'][:].data
        lon = op['lon'][:].data
        lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],op['time'].shape[0],1),order='F')
        lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],op['time'].shape[0],1),order='F')
        time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 
        
        latoutAl = np.append(latoutAl,np.ndarray.flatten(lat_out))
        lonoutAl = np.append(lonoutAl,np.ndarray.flatten(lon_out))
        tout = np.append(tout,np.ndarray.flatten(time_out))
        
        
print('...done...')

newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/GEFS_out/'
if not os.path.exists(newdir):
        os.makedirs(newdir)

print('IVTm shape:', IVTm_pp_gefs.shape)
print('IVT mean shape:', IVT_ppm_gefs.shape)
print('IVT std shape:', IVT_ppv_gefs.shape)


df_out = pd.DataFrame({'time':tout,'OBS':IVTm_pp_gefs,'IVTmean':IVT_ppm_gefs,'IVTstd':np.sqrt(IVT_ppv_gefs),'lat':latoutAl,'lon':lonoutAl})
df_out.to_pickle(newdir+'Mean_Variance_PP_gefs.pkl')


np.savez(newdir+'Mean_Variance_PP_gefs.npz',lat=latgefs,lon=longefs,IVTgefsm=IVT_ppm_gefs,IVTgefss=np.sqrt(IVT_ppv_gefs),IVTm_gefs=IVTm_pp_gefs)

print('\n')
print('\n')
print('\n')
print('\n')
print('\n')


print('#############################################################################')
print('#############################################################################')
print('I saved this file:')
print(newdir+'Mean_Variance_PP_gefs.pkl')
print('#############################################################################')
print('############## *I hope that is okay baby* - gale beggy ######################')
print('#############################################################################')

print(df_out)
print(df_out.isnull().sum())
print(df_out.describe())


print('#############################################################################')
print('############## *I hope that is okay baby* - gale beggy ######################')
print('#############################################################################')