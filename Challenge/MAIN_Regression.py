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
   
test['tH2_pred'] = sol
s.soumission(test)
