# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 12:43:25 2021

@author: Tasbiha
"""


from pyts.datasets import fetch_uea_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
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

from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test =fetch_uea_dataset('JapaneseVowels', use_cache=True, data_home=None, return_X_y=True)



print(uea_dataset_info('JapaneseVowels'))

encoder = Encoder(10, 0.9, 0, 255)
esnn = ESNN(encoder, m=0.9, c=0.75, s=0.03)
esnn.train(X_train, y_train)
    
y_pred = esnn.test(X_test)

acc = accuracy_score(y_test, y_pred)
auc= auc(y_test, y_pred)
conf=confusion_matrix(y_test, y_pred)  
pre = precision_score(y_test, y_pred, average ='weighted',zero_division = 1) 
re =recall_score(y_test, y_pred, average ='weighted', zero_division=1) 
FScore=f1_score(y_test, y_pred, average = 'weighted')  
print(f"Neuron Count: {len(esnn.all_neurons)}")
print(f"Accuracy: {acc}")
print(f"Auc: {auc}")
print(f"Confusion Metrics: {conf}")
print(f"Precision: {pre}")
print(f"Recall: {re}")
print(f"F1_score: {FScore}")


