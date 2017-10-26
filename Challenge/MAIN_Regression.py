#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Main

import pandas as pd
import pre_processing as pre
from sklearn import linear_model
import Soumission as s 

#######################################################
#          PARTIE D'EVALUATION DU MODELE              #
#  (Tester notre modèle sans attendre les 1h du site) #
#######################################################

dataset = pd.read_csv('./Datasets/PP_train.csv', sep=";", decimal = ",")
d = pre.DataSet(dataset)
reg = linear_model.LinearRegression(normalize = True)
reg.fit (d.X_train, d.y_train)


sol = reg.predict(d.X_test)
sol2 = reg.predict(d.X_train)
print(s.RMSE(d.y_test,sol))
print(s.RMSE(d.y_train,sol2))
LinearRegression(dummyEch,dropNA,dropColNATest)

#test = test.dropna(axis=1)


#######################################################################
#      PARTIE DE CREATION DU MODELE FINAL (AVEC TOUTES LES DONNEES)   #
#               ET DE PREDICTION AVEC LE JEU DE TEST DONNE            #
#######################################################################

train = pd.read_csv('./Datasets/PP_train.csv', sep=";", decimal = ",")
test = pd.read_csv('./Datasets/PP_test.csv', sep=";", decimal = ",")



d = pre.DataSet(train,test)

reg = linear_model.LinearRegression(normalize = True)
reg.fit (d.X_train, d.y_train)

sol = reg.predict(d.X_test)
sol2 = reg.predict(d.X_train)
print(s.RMSE(d.y_train,sol2))

   
test['tH2_pred'] = sol
s.soumission(test)

#######################################################################
#                         Apprentissage par années                     # 
#                                                                      #
#######################################################################

mask = train['date'].str.contains("2014")
df14 = train[mask]
df15 = train[~mask]

X_train = df14[train.columns.difference(["tH2_obs","date"])]
y_train = df14["tH2_obs"]
X_test = df15[train.columns.difference(["tH2_obs","date"])]
y_test = df15["tH2_obs"]


reg = linear_model.LinearRegression(normalize = True)
reg.fit (X_train, y_train)

y_pred_test = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
print(s.RMSE(y_test,y_pred_test)) # gros ecart 1,5 possible explication de nos res 2016
print(s.RMSE(y_train,y_pred_train))


#######################################################################
#                         Cross validation spliting                    # 
#                                                                      #
#######################################################################

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from sklearn import metrics 


d = pre.DataSet(train,test)
lm = linear_model.LinearRegression()


#model = lm.fit(d.X_train, d.y_train)

sol = lm.predict(d.X_test)
sol2 = lm.predict(d.X_train)
print(s.RMSE(d.y_test,sol))
print(s.RMSE(d.y_train,sol2))

test['tH2_pred'] = sol
s.soumission(test)


# 2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

d = pre.DataSet(train,test)
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(d.X_train)
X_test_ = poly.fit_transform(d.X_test)

lg = LinearRegression()

# Fit
lg.fit(X_, d.y_train)

# Obtain coefficients

y = lg.predict(X_test_)
y2 = lg.predict(X_)

test['tH2_pred'] = y
s.soumission(test)

print(s.RMSE(d.y_train,y2))

# sur training 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

d = pre.DataSet(train)
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(d.X_train)
X_test_ = poly.fit_transform(d.X_test)

lg = LinearRegression()

# Fit
lg.fit(X_, d.y_train)

y = lg.predict(X_test_)
y2 = lg.predict(X_)

test['tH2_pred'] = y
s.soumission(test)
print(s.RMSE(d.y_test,y))
print(s.RMSE(d.y_train,y2))

#1
#faire acp sur l'ecrat pour inferer
from sklearn.preprocessing import StandardScaler

d = pre.DataSet(train,test)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(d.X_train)
y = sc_y.fit_transform(d.y_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)

