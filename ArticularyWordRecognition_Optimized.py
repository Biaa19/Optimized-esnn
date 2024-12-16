# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:14:00 2021

@author: Tasbiha
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:13:44 2021

@author: Tasbiha
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:39:10 2021

@author: Tasbiha
"""
from pyts.datasets import fetch_uea_dataset
from sklearn.datasets import  load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from SalpSwarm.SalpSwarm import *
from esnn.esnn import ESNN
from esnn.encoder import Encoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc

acclist=[]
clist=[]
mlist=[]
slist=[]
neuronlist=[]
recalllist=[]
precisionlist=[]
f1list=[]
auclist=[]
y_predlist=[]
y_testlist=[]
y_trainlist=[]
X_trainlist=[]
X_testlist=[]
X_train, X_test, y_train, y_test =fetch_uea_dataset('ArticularyWordRecognition', use_cache=True, data_home=None, return_X_y=True)


for x in range(0,50):
 
   _m =  salp_swarm_algorithm(swarm_size = 80, min_values = [1.0,1.0], max_values = [0.90,0.90], iterations = 500, target_function = rosenbrocks_valley)
   _c = salp_swarm_algorithm(swarm_size = 80, min_values = [0.80,0.80], max_values = [0.50,0.50], iterations = 500, target_function = rosenbrocks_valley)
   _s = salp_swarm_algorithm(swarm_size = 80, min_values = [0.09, 0.09], max_values = [0.02,0.02], iterations = 500, target_function = rosenbrocks_valley)
  # _m = ant_lion_optimizer(colony_size = 80, min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
   #_c= ant_lion_optimizer(colony_size = 80, min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
   #_s = ant_lion_optimizer(colony_size = 80, min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
   #_m =  whale_optimization_algorithm(hunting_party = 100, spiral_param = 0.5,  min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
   #_c =  whale_optimization_algorithm(hunting_party = 100, spiral_param = 0.5,  min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
   #_s =  whale_optimization_algorithm(hunting_party = 100, spiral_param = 0.5,  min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
   _m = _m.reshape(-1)[0]
   _c = _c.reshape(-1)[0]
   _s = _s.reshape(-1)[0]

   if _m >0 and _m < 1:
       if _c > 0 and _c < 1:
           if _s >0 and _s <1:
                
               encoder = Encoder(10, 1, 1, 255)
               esnn = ESNN(encoder, m=_m, c=_c, s=_s)
               esnn.train(X_train, y_train)
               y_pred = esnn.test(X_test)
               acc =accuracy_score(y_test, y_pred) 
               FScore=f1_score(y_test, y_pred, average = 'weighted')  
               pre = precision_score(y_test, y_pred, average ='weighted', zero_division= 1) 
               re =recall_score(y_test, y_pred, average ='weighted', zero_division= 1) 
               #auc= auc(y_test, y_pred)
               neuronlist.append(len(esnn.all_neurons))
               acclist.append(acc)
               mlist.append(_m)
               slist.append(_s)
               clist.append(_c)
               precisionlist.append(pre)
               recalllist.append(re)
               f1list.append(FScore)
               auclist.append(auc)
               y_predlist.append(y_pred)
               y_testlist.append(y_test)
               y_trainlist.append(y_train)
               
                 
rounded_acclist = [ round(elem, ) for elem in acclist ] 
rounded_mlist = [ round(elem, 2) for elem in mlist ] 
rounded_slist = [ round(elem, 2) for elem in slist ] 
rounded_clist = [ round(elem, 2) for elem in clist ] 
rounded_relist=[round(elem, 2) for elem in recalllist]
rounded_precisionlit=[round(elem, 2) for elem in  precisionlist ]
rounded_flist =[round(elem, 2) for elem in f1list]
#rounded_auclist=[round(elem, 2) for elem in auclist]
 
print(f"Mod: {mlist}")
print(f"Sim: {slist}")
print(f"Threshold: {clist}")
print(f"ACC Average: {statistics.mean(acclist)}")
#print(f"Auc: {rounded_auclist}")
print(f"Neuron Count: {len(esnn.all_neurons)}")
#print(f"Accuracy: {acc}")
print(f"Accuracy: {acclist}")
print(f"F1_score: {f1list}")
print(f"Precision: {precisionlist}")
print(f"Recall: {recalllist}")
#print(f"Y_pred: {y_predlist}")
#print(f"Y_test: {y_testlist}")




## convert your array into a dataframe
df1 = pd.DataFrame (y_predlist)

## save to xlsx file

filepath1 = 'ArticularyWordRecognition ypred.xlsx'

df1.to_excel(filepath1, index=True)
df2 = pd.DataFrame (y_testlist)

## save to xlsx file

filepath2 = 'ArticularyWordRecognition ytest.xlsx'

df2.to_excel(filepath2, index=True)