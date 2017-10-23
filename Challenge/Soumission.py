#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing as p
import sklearn.model_selection as sk


def soumission(res,frametype = 'All'): #En entrée, il faut le jeu de test complétée avec la valeur prédite (Avec comme nom tH2_pred)
    res = res.loc[:,('date', 'insee', 'ech', 'tH2_pred')] #On récupère uniquement les éléments à soummettre
    template = pd.read_csv('./Datasets/test_answer_template.csv', sep=";", decimal = ",")
    soumission = pd.merge(template, res, on=['date','insee', 'ech']) #On fait une jointure avoir exactement le même format que le template
    soumission["tH2_obs"] = soumission["tH2_pred"]
    soumission = soumission.loc[:,('date', 'insee', 'ech', 'tH2_obs')] 
    fileName = input("Nom du fichier? \n")
    soumission.to_csv(fileName +'.csv',sep=";", decimal = ",",index = False)

def evaluate(classifier, X, y): # (X,y) is a testing set
    return(mean(classifier.predict(X) != y))
    
    
def RMSE(y_actual, y_predicted):
    return(sqrt(mean_squared_error(y_actual, y_predicted)))
    

            
