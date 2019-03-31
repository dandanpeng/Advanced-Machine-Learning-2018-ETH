#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:07:22 2018

@author: pengdandan
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import preprocessing

x_train = pd.read_csv('X_train.csv')
x_train = x_train.iloc[:,1:]

y_train = pd.read_csv('y_train.csv')
y_train = y_train.iloc[:,1]
         
x_test = pd.read_csv('X_test.csv')
id = pd.DataFrame(x_test['id'])
x_test = x_test.iloc[:,1:]

x_train = StandardScaler().fit_transform(x_train)
x_train = MinMaxScaler().fit_transform(x_train)

x_test = StandardScaler().fit_tranform(x_test)

x_train = VarianceThreshold(threshold = 0.2).fit_transform(x_train)
var = np.var(x_train,axis = 1)

x_train['label'] = y_train
one = x_train[x_train.label == 1].sample(1000,replace = True)
zero = x_train[x_train.label == 0].sample(1000,replace = True)
two = x_train[x_train.label == 2]
downsample = one.sample(600)
data = pd.concat([downsample,zero,two])
x_train = data.iloc[:,1:-1]
y_train = data.iloc[:,-1]

class_weight = dict({0:5.5,1:1,2:5.5})
clf = SVC(class_weight = 'balanced',decision_function_shape = 'ovo',gamma = 'scale')

prediction = pd.DataFrame(prediction)
id['y'] = prediction
id.to_csv('prediction9.csv',index = False)


train,test,y_train,y_test = train_test_split(x_train,y_train,test_size = 0.1,random_state =0)
clf.fit(train,y_train)
prediction = clf.predict(x_test)
balanced_accuracy_score(y_test,prediction)
