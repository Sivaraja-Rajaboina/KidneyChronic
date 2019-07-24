#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:43:38 2019

@author: z003vhn
"""

import pandas as pd
import numpy as np



dataset = pd.read_csv("kidneyChronic.csv")

print(dataset.describe())
print(dataset.isnull().sum())

print(type(dataset))
dataset = dataset.replace(to_replace = "\?" ,value=np.nan,regex =True)

print(dataset.isnull().sum())

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,24].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputerMode = SimpleImputer(strategy="most_frequent")
x[:,0:5] = imputer.fit_transform(x[:,0:5])
x[:,5:9] = imputerMode.fit_transform(x[:,5:9])
x[:,9:18] = imputer.fit_transform(x[:,9:18])
x[:,18:24] = imputerMode.fit_transform(x[:,18:24])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
x[:, 5] = labelencoder_X.fit_transform(x[:, 5])
x[:, 6] = labelencoder_X.fit_transform(x[:, 6])
x[:, 7] = labelencoder_X.fit_transform(x[:, 7])
x[:, 8] = labelencoder_X.fit_transform(x[:, 8])
x[:, 18] = labelencoder_X.fit_transform(x[:, 9])
x[:, 19] = labelencoder_X.fit_transform(x[:, 19])
x[:, 20] = labelencoder_X.fit_transform(x[:, 20])
x[:, 21] = labelencoder_X.fit_transform(x[:, 21])
x[:, 22] = labelencoder_X.fit_transform(x[:, 22])
x[:, 23] = labelencoder_X.fit_transform(x[:, 23])

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 0)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)

from sklearn.metrics import accuracy_score

print("accuracy    ",accuracy_score(y_test,y_pred))