import pandas as pd
import numpy as np
import os
import sys

desired_year = sys.argv[1]
df_t = pd.read_csv('./ForecastIssueDate.txt',names=['Issued_date'])
df_t['year'] = [str(bb)[:4] for bb in df_t['Issued_date']]
df_t['month'] = [str(bb)[4:6] for bb in df_t['Issued_date']]
df_t['day'] = [str(bb)[6:8] for bb in df_t['Issued_date']]
indexfirst = int((df_t[(df_t['year']==desired_year) & (df_t['month']=='12')].index[0] / 21)+1)
print(indexfirst)

##############################
svfil = './starting_index.txt'
with open(svfil, 'w') as f:
    f.write('%d' % indexfirst)
