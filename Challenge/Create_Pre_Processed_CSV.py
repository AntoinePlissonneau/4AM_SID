#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:57:19 2017

@author: AntoineP
"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


        
def import_train() :
    dataset = pd.DataFrame()       
    for i in range(1,37): 
        data = pd.read_csv('./Datasets/train_'+str(i)+'.csv', sep=";", decimal = ",", encoding = "utf-8")
        dataset = dataset.append(data,ignore_index=True)
    return dataset

  
def to_float(dataset) :
    for i in dataset.iloc[:,1:dataset.shape[1]-1].columns:
        if type(dataset[i][0]) == str :
            dataset[i].replace(",",".",regex=True,inplace=True)
            dataset[i] = dataset[i].astype(np.float64)
        if type(dataset[i][0]) == float :
            dataset[i] = np.float64(dataset[i])
    dataset["huH2"] = dataset["huH2"].values / 100
    return dataset

    
def to_date(dataset):
    
    dataset['date'] = pd.to_datetime(dataset['date'])
    return dataset
    
def to_dummy(dataset,colname) :
    # Attention au trap, on drop une dummy
    onehot = pd.get_dummies(dataset[colname], drop_first = False)
    onehot = onehot.rename(columns=lambda x: colname + '_' + str(x))
    dataset = dataset.join(onehot)    
    return dataset
    
    # Dealing with missing data
    #l'objet groupby se compose de key et de l'item : key correspond a ce que l'on a groupé
    #alors que l'item est un sous dataframe issue de notre key.

def missing_data_mean(dataset):
    colonne = dataset.iloc[:,1:43].columns.difference(["insee","mois"])
    for col in colonne :
        dataset[col] = dataset.iloc[:,1:43].groupby(['insee','mois']).transform(lambda x: x.fillna(x.mean()))[col]
    return dataset
        
            
def drop(dataset,colname):
    
    dataset = dataset.drop(colname, axis = 1)
    
    return dataset

def to_csv(dataset, filename):
    dataset.to_csv("./Datasets/"+ filename +'.csv',sep=';',index=False, encoding = "utf-8",decimal = ",")
    return dataset
        
    




        

    
# Preprocessing sur le train


train = import_train()
train = to_float(train)
#train = to_date(train)
train = prp.to_dummy(train,"insee")
train = to_dummy(train,"mois")
train = missing_data_mean(train)
train = drop(train,"mois")
to_csv(train,"PP_train")

# Preprocessing sur le test

test = pd.read_csv("./Datasets/test.csv", sep = ";", decimal = ","  , encoding = "utf-8")
test = to_float(test)
#test = to_date(test)
test = prp.to_dummy(test,"insee")
test = to_dummy(test,"mois")
test = missing_data_mean(test)
test = drop(test,"mois")
test["mois_janvier"] = 0
test["mois_février"] = 0
test["mois_mars"] = 0
test["mois_avril"] = 0
test = test.loc[:,('date', 'insee', 'capeinsSOL0', 'ciwcH20', 'clwcH20',
       'ddH10_rose4', 'ffH10', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0',
       'flvis1SOL0', 'hcoulimSOL0', 'huH2', 'iwcSOL0', 'nbSOL0_HMoy', 'nH20',
       'ntSOL0_HMoy', 'pMER0', 'rr1SOL0', 'rrH20', 'tH2', 'tH2_VGrad_2.100',
       'tH2_XGrad', 'tH2_YGrad', 'tpwHPA850', 'ux1H10', 'vapcSOL0', 'vx1H10',
       'ech', 'insee_31069001', 'insee_33281001', 'insee_35281001',
       'insee_59343001', 'insee_67124001', 'insee_75114001', 'mois_août',
       'mois_avril', 'mois_décembre', 'mois_février', 'mois_janvier',
       'mois_juillet', 'mois_juin', 'mois_mai', 'mois_mars', 'mois_novembre',
       'mois_octobre', 'mois_septembre')]
to_csv(test,"PP_test")

# Scaling

#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(d.X_train)
#X_test = sc_X.transform(d.X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(d.y_train)


