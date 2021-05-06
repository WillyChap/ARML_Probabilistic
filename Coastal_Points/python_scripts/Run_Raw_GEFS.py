

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

latoutAl = np.array([])
lonoutAl = np.array([])
tout = np.array([],dtype='datetime64[ns]')

print('GEFS FILE NAMES:',gefs_file_names)

for aa,f in enumerate(gefs_file_names):
    ff = Dataset(f)
    
    if aa==0:
        IVTgefs = ff['IVT'][:].data
        IVTm_gefs = ff['IVTm'][:].data
        latgefs = ff['lat'][:]
        longefs = ff['lon'][:]
        
        #gather other variables of interest (time and lat/lon)
        op = xr.open_dataset(f)
        lat = op['lat'][:].data
        lon = op['lon'][:].data
        lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],IVTgefs.shape[0],1),order='F')
        lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],IVTgefs.shape[0],1),order='F')
        time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 
        
        IVTgefsm = (np.squeeze(np.mean(IVTgefs,axis=1)))
        IVTgefss = (np.squeeze(np.std(IVTgefs,axis=1)))
        IVTgefsm = np.ndarray.flatten(IVTgefsm[:,latind,lonind],order='F')
        IVTgefss = np.ndarray.flatten(IVTgefss[:,latind,lonind],order='F')
        IVTm_gefs = np.ndarray.flatten(IVTm_gefs[:,latind,lonind],order='F')
        
        latoutAl = np.append(latoutAl,np.ndarray.flatten(lat_out))
        lonoutAl = np.append(lonoutAl,np.ndarray.flatten(lon_out))
        tout = np.append(tout,np.ndarray.flatten(time_out))
        
        
    else:
        IVTgefsT = ff['IVT'][:].data
        IVTm_gefsT = ff['IVTm'][:].data
        latgefs = ff['lat'][:]
        longefs = ff['lon'][:]
        
        #gather other variables of interest (time and lat/lon)
        op = xr.open_dataset(f)
        lat = op['lat'][:].data
        lon = op['lon'][:].data
        lat_out = np.ndarray.flatten(np.matlib.repmat(lat[latind],IVTgefsT.shape[0],1),order='F')
        lon_out = np.ndarray.flatten(np.matlib.repmat(lon[lonind],IVTgefsT.shape[0],1),order='F')
        time_out = np.ndarray.flatten(np.matlib.repmat(op['time'][:].data,latind.shape[0],1)) 
        
        IVTgefsmT = (np.squeeze(np.mean(IVTgefsT,axis=1)))
        IVTgefssT = (np.squeeze(np.std(IVTgefsT,axis=1)))
        IVTgefsmT = np.ndarray.flatten(IVTgefsmT[:,latind,lonind],order='F')
        IVTgefssT = np.ndarray.flatten(IVTgefssT[:,latind,lonind],order='F')
        IVTm_gefsT = np.ndarray.flatten(IVTm_gefsT[:,latind,lonind],order='F')

        IVTgefsm = np.concatenate([IVTgefsm, IVTgefsmT],axis=0)
        IVTgefss = np.concatenate([IVTgefss, IVTgefssT],axis=0)
        IVTm_gefs = np.concatenate([IVTm_gefs,  IVTm_gefsT],axis=0)
        
        latoutAl = np.append(latoutAl,np.ndarray.flatten(lat_out))
        lonoutAl = np.append(lonoutAl,np.ndarray.flatten(lon_out))
        tout = np.append(tout,np.ndarray.flatten(time_out))
        
        
newdir = '/glade/scratch/wchapman/Reforecast/models/NN_CRPS/' +fcast+'/GEFS_out/'
if not os.path.exists(newdir):
        os.makedirs(newdir)
        
df_out = pd.DataFrame({'time':tout,'OBS':IVTm_gefs,'IVTmean':IVTgefsm,'IVTstd':IVTgefss,'lat':latoutAl,'lon':lonoutAl})
df_out.to_pickle(newdir+'RAW_gefs.pkl')


np.savez(newdir+'RAW_gefs.npz',lat=latgefs,lon=longefs,IVTgefsm=IVTgefsm,IVTgefss=IVTgefss,IVTm_gefs=IVTm_gefs)

print('.done.')
print(df_out)
print(df_out.isnull().sum())
print(df_out.describe())

print('#############################################################################')
print('############## *I hope that is okay baby* - gale beggy ######################')
print('#############################################################################')