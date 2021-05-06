import numpy as np
import pandas as pd 
import os
import sys
import  csv

num2 = pd.read_csv('./starting_index.txt',names=['num'])['num'][0];
line1 = [21, 21, num2, 4001, 4002, 4122] #forecast lead times, ensembles, train_start, train_end, test_start, test_end 
line2 = [6] #number of predictors 
nnm = [f'{x:03}' for x in range(line2[0])]
FilO = pd.read_csv('./weight.txt',delim_whitespace=True,names=nnm)
#FilO = pd.read_csv('./weight_IVT.txt',delim_whitespace=True,names=nnm)
line3 = FilO.iloc[0,:].tolist()
line4 = [1, 1, 1, 1, 1, 1]
line5 = [300.] #bias



myList = [line1,
         line2,
         line3,
         line4,
         line5]


writepath = './analog_ini.txt'


try:
    os.remove(writepath)
except OSError:
    pass

print('writing list')
with open(writepath,"w") as f:
    wr = csv.writer(f,delimiter=" ")
    wr.writerows(myList)
