# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 12:43:27 2021

@author: Tasbiha
"""


from pyts.datasets import fetch_uea_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from esnn.esnn import ESNN
from esnn.encoder import Encoder
from pyts.datasets import ucr_dataset_info
from pyts.datasets import ucr_dataset_list
ucr_dataset_list()[:30]
ucr_dataset_info()
from pyts.datasets import uea_dataset_info
uea_dataset_info()

X_train, X_test, y_train, y_test =fetch_uea_dataset('ArticularyWordRecognition', use_cache=True, data_home=None, return_X_y=True)



encoder = Encoder(10, 0.9, 0, 255)
esnn = ESNN(encoder, m=0.9, c=0.75, s=0.01)
esnn.train(X_train, y_train)

    
y_pred = esnn.test(X_test)

acc = accuracy_score(y_test, y_pred)
#roc= roc_auc_score(y_test, y_pred, multi_class='ovo')
FScore=f1_score(y_test, y_pred, average = 'weighted')  
pre = precision_score(y_test, y_pred, average ='weighted', zero_division= 1) 
re =recall_score(y_test, y_pred, average ='weighted', zero_division= 1) 
print(f"Neuron Count: {len(esnn.all_neurons)}")
print(f"Accuracy: {acc}")
#print(f"ROC AUC: {roc}")
print(f"F1_score: {FScore}")
print(f"Precision: {pre}")
print(f"Recall: {re}")