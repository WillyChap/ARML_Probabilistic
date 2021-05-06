import numpy as np
import pandas as pd 
import os
import sys


if os.path.exists(sys.argv[2] +'AnEnVarsClean.txt'):
    os.remove(sys.argv[2] +'AnEnVarsClean.txt')
FilO = pd.read_csv(sys.argv[1],sep=' ',names=['IVT','SLP','U500','V500','Z500','IWV','IVTm','Cat'])
FilO = FilO.drop(['Cat'], axis=1)

for bb in FilO:
    if np.sum(np.isnan(FilO[bb])) > 0:
        FilO.loc[np.where(np.isnan(FilO[bb]))]=-9999

FilO.loc[np.where((FilO['Z500'])==0)] = -9999

FilO = FilO.astype(object)
FilO.loc[FilO['IVT'] == '--'] =-9999

writepath = sys.argv[2] +'AnEnVarsClean.txt'
FilO.to_csv(writepath, header=None, index=None, sep=' ', mode='a')
