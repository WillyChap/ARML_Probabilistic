import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import os
from netCDF4 import Dataset,num2date
import matplotlib.pyplot as plt
import glob
import sys


def Create_Arrays_AnEn_Model(AnEn_Path_This_Decade_This_Season,mmnn,Transformed,fill,unico,Output,analog_ini):

    TEST = pd.DataFrame()
    EPS_ANALOG = pd.DataFrame()
    MODEL = pd.DataFrame()
        
    for this_st_path in glob.glob(AnEn_Path_This_Decade_This_Season):
               
        Analogs_Dict = dict()
        for i in np.arange(mmnn):
            Analogs_Dict['Analog'+str(i)] = []
            
        for i in range(len(Output)):
            line = Output[Output.columns[0]].values[i].split()
            line = [float(z) for z in line]
            for j in range(len(line)):
                Analogs_Dict['Analog'+str(j)].append(line[j])
        
        
        eps_analog = pd.DataFrame.from_dict(Analogs_Dict)

        if Transformed == True:
            for cc in eps_analog.columns:
                eps_analog[cc] = eps_analog[cc] ** 3 

        
        
        test_start = (analog_ini.at[0,3]) * analog_ini.at[0,0]
        num_predictors = len(unico.columns)-1
        num_analogs = analog_ini.at[0,1]
        

        unico_output = unico.loc[unico.index>=test_start]
        
        this_station_test = pd.DataFrame()    
        this_station_test ['LT'] = np.tile(np.arange(analog_ini.at[0,0]),               int(len(unico_output)/analog_ini.at[0,0]))
        
        this_station_test ['OBS'] = unico_output[unico_output.columns[-1]].values
        if Transformed == True:
            this_station_test ['OBS'] = this_station_test ['OBS'] ** 3
            
    
        
        this_station_model = pd.DataFrame()
        this_station_model['Model'] = unico_output[unico_output.columns[0]].values

        if Transformed == True:
            this_station_model ['Model'] = this_station_model ['Model'] ** 3
            
        TEST = TEST.append(this_station_test)
        EPS_ANALOG = EPS_ANALOG.append(eps_analog)
        MODEL = MODEL.append(this_station_model)

    ##==========================================================================##
    
    TEST[TEST<-1000] = fill
    EPS_ANALOG[EPS_ANALOG<-1000] = fill
    MODEL[MODEL<-1000] = fill     


    return [TEST, EPS_ANALOG, MODEL]

yrz = sys.argv[1]
AnEnFile = sys.argv[2]
OutFile = sys.argv[3]
cdd = sys.argv[4]
noense = int(sys.argv[5])
strtyear = sys.argv[6]

unico = pd.read_csv(AnEnFile,header = None, delimiter = ' ')
Output = pd.read_csv(OutFile,header = None)
analog_ini = pd.read_csv(cdd+'/analog_ini.txt',header = None, delimiter = ' ')
[OBS,Analogs,Model_Pred]=Create_Arrays_AnEn_Model(cdd,noense,False,0,unico,Output,analog_ini)
svDF = pd.concat([OBS,Model_Pred,Analogs],axis=1)

unico = pd.read_csv(AnEnFile,header = None, delimiter = ' ')
Output = pd.read_csv(OutFile,header = None)
analog_ini = pd.read_csv(cdd+'/analog_ini.txt',header = None, delimiter = ' ')
[OBS,Analogs,Model_Pred]=Create_Arrays_AnEn_Model(cdd,noense,False,0,unico,Output,analog_ini)
svDF = pd.concat([OBS,Model_Pred,Analogs],axis=1)
# print(svDF.shape)
# svfil = cdd+'/AnEn_' + str(yrz) +'.pkl'
# svDF.to_pickle(svfil)

TimeForIss=[]
with open('./ForecastIssueDate.txt', "r") as f:
    for line in f:
        TimeForIss.append(line.strip())
        
bb=[datetime.strptime(f1,'%Y%m%d%H') for f1 in TimeForIss]
forstr = [ 'F'+f'{y:03}' for x in range(int(len(TimeForIss)/len(np.arange(0,126,6)))) for y in np.arange(0,126,6)]

fpFILES = '/data/downloaded/Forecasts/Machine_Learning/Reforecast/buildAnEn/AllFiles_F120.txt'

filesAll=[]
with open(fpFILES, "r") as f:
    for line in f:
        filesAll.append(line.strip())

alldict = {'Fname':filesAll,'Issued_Time': bb, 'lead_hr': forstr}
df_new=pd.DataFrame(alldict)
maskin  = (df_new['Issued_Time'] > yrz+'-11-1') & (df_new['Issued_Time'] <= str(int(yrz)+1)+'-5-10')
df_new = df_new.loc[maskin]
df_new=df_new.reset_index(drop=True)

svDF = pd.concat([df_new, svDF], axis=1, sort=False)

print(svDF.shape)
svfil = cdd+'/AnEn_' + str(yrz) +'_21Ense_StartingYear_' + strtyear + '.pkl'
svDF.to_pickle(svfil)
