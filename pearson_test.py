# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:07:13 2021

@author: Miran
"""
import pandas as pd
import numpy as np
import glob
from scipy import stats
from sklearn.metrics import mean_absolute_error

cnt = 0

personalID = "test"

pspi_dir = "./data/pearson/"+personalID+"/pspi/*.xlsx"
int_dir = "./data/pearson/"+personalID+"/intensity/*.xlsx"

for f in glob.glob(pspi_dir):
   cnt=cnt+1
   df = pd.read_excel(f, header=None)
   # print(df)
   tmp = df[df.columns[0:]].to_numpy()
   if(cnt ==1 ):
       all_data = tmp
   else: 
       all_data = np.append(all_data, tmp)

cnt = 0
for f in glob.glob(int_dir):
   cnt=cnt+1
   inr_df = pd.read_excel(f, header=None)
   # print(df)
   int_tmp = inr_df[inr_df.columns[0:]].to_numpy()
   if(cnt ==1 ):
       int_data = int_tmp
   else: 
       int_data = np.append(int_data, int_tmp)

A = all_data
B = int_data 

print(stats.pearsonr(A,B))
print(mean_absolute_error(A, B))

