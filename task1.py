#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:09:49 2018

@author: pengdandan
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import Imputer
from sklearn.linear_model import Lasso,Ridge
from sklearn.metrics import r2_score
x_train = pd.read_csv("data1_wo.csv")
y_train = x_train.iloc[:,-1]
x_train = x_train.iloc[:,1:-1]
x_test = pd.read_csv("test1.csv")
ID = pd.DataFrame(x_test['id'])
x_test = x_test.iloc[:,1:]


imp_mean = Imputer(missing_values = np.nan,strategy = 'median')
imp_mean.fit(x_train)
imp_x_train = imp_mean.transform(x_train)

clf = IsolationForest(max_samples = 1100,max_features = 400)
clf.fit(imp_x_train)
train = clf.predict(imp_x_train).tolist()
[train.index(train) for i in train if train == -1]

k = 10
num_val_samples = len(x_train) // k
all_score = []
for i in range(k):
    print('processing fold #',i)
    val_data = x_train[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = y_train[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data = np.concatenate(
            [x_train[:i*num_val_samples],
             x_train[(i+1)*num_val_samples:]],
             axis =0)
    partial_train_targets = np.concatenate(
            [y_train[:i*num_val_samples],
             y_train[(i+1)*num_val_samples:]],
             axis = 0)
    model = Lasso(alpha = 0.01)
    model.fit(partial_train_data,partial_train_targets)
    #val_mse,val_mae = model.evaluate(val_data,val_targets,verbose = 0)
    prediction = model.predict(x_train[i*num_val_samples+1:(i+1)*num_val_samples-1])
    score = r2_score(y_train[i*num_val_samples+1:(i+1)*num_val_samples-1],prediction)
    all_score.append(score)
np.mean(all_score)

prediction = model.predict(x_test)
ID['y'] = prediction
ID.to_csv('prediction21.csv',index = False)


sigma = np.cov(imp_x_train)
norm_x_train = imp_x_train - np.mean(imp_x_train,axis = 0)
sigma = np.cov(norm_x_train)
np.linalg.eig(sigma)
