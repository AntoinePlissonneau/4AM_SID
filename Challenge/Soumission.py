#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:04:44 2017

@author: AntoineP
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing as p
import sklearn.model_selection as s




def soumission(res,frametype = 'All'): #En entrée, il faut le jeu de test complétée avec la valeur prédite (Avec comme nom tH2_pred)
    res = res.loc[:,('date', 'insee', 'ech', 'tH2_pred')] #On récupère uniquement les éléments à soummettre
    template = pd.read_csv('./Datasets/test_answer_template.csv', sep=";", decimal = ",")
    soumission = pd.merge(template, res, on=['date','insee', 'ech']) #On fait une jointure avoir exactement le même format que le template
    soumission["tH2_obs"] = soumission["tH2_pred"]
    soumission = soumission.loc[:,('date', 'insee', 'ech', 'tH2_obs')] 
    fileName = input("Nom du fichier? \n")
    soumission.to_csv(fileName +'.csv',sep=";", decimal = ",",index = False)
    
    
    
    
#Test où la température prédite est celle du modèle déterministe
#test = pd.read_csv('./Datasets/test.csv', sep=";", decimal = ",")
#test['tH2_pred'] = test['tH2']
#soumission(test)


def evaluate(classifier, X, y): # (X,y) is a testing set
    return(mean(classifier.predict(X) != y))
    
    
def RMSE(y_actual, y_predicted):
    return(sqrt(mean_squared_error(y_actual, y_predicted)))
    
    
class DataSet:
    def __init__(self,train,test = 'ok'):
        X = train[train.columns.difference(["tH2_obs","date"])]
        y = train["tH2_obs"]
        
        if type(test) != pd.pandas.core.frame.DataFrame:
            X_train, X_test, y_train, y_test = s.train_test_split(X, y, test_size = 0.2, random_state = 1)
            self.X_train = X_train # input : images
            self.X_test = X_test # default: features = data itself
            self.y_train = y_train # output: labels
            self.y_test = y_test # output: labels
            self.n = len(train)
        
        else:
            self.X_train = X
            self.y_train = y
            self.X_test = test[test.columns.difference(["date"])]
            
