# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:57:41 2017

@author: Amira AYADI
"""

import numpy as np 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importing the dataset

dataset = pd.read_csv('./Dataset/train_1.csv', sep=";", decimal = ",")
dataset = pd.read_csv('./Dataset/tes.csv', sep=";")

dataset["ddH10_rose4"] = dataset["ddH10_rose4"].astype(np.float64)

#dataset = pd.concat((train_1, train2), axis=0, ignore_index=True)

# Dealing with the date
dataset['date'] = pd.to_datetime(dataset['date'])

dataset['ordinal_date']=dataset['date'].map(dt.datetime.toordinal)
dataset["year"] = dataset['date'].dt.year
dataset["month"] = dataset['date'].dt.month
dataset["day"] = dataset['date'].dt.day
dataset["weekofyear"] = dataset['date'].dt.weekofyear

dataset.drop('date', axis=1, inplace=True)
dataset.drop('mois', axis=1, inplace=True)


# Dealing with missing data
#l'objet groupby se compose de key et de l'item : key correspond a ce que l'on a groupé
#alors que l'item est un sous dataframe issue de notre key.
colonne = dataset.columns.difference(["insee","month"])
for col in colonne :
    dataset[col] = dataset.groupby(['insee','month']).transform(lambda x: x.fillna(x.mean()))[col]

# X and y 
X = dataset[dataset.columns.difference(["tH2_obs","ordinal_date"])].values
y = dataset["tH2_obs"].values


    
