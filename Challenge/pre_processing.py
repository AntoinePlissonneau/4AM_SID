# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:57:41 2017

@author: Amira AYADI

"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessing:

    #rajouter init class
    
    def importing_data(self,dataset) :
        
        for i in range(1,37): 
            data = pd.read_csv('./Datasets/train_'+str(i)+'.csv', sep=";", decimal = ",", encoding = "utf-8")
            dataset = dataset.append(data,ignore_index=True)
        return dataset
    
    def to_float_train(self,dataset) :
        for i in dataset.iloc[:,1:30].columns:
            if type(dataset[i][0]) == str :
                dataset[i].replace(",",".",regex=True,inplace=True)
                dataset[i] = dataset[i].astype(np.float64)
            if type(dataset[i][0]) == float :
                dataset[i] = np.float64(dataset[i])
        dataset["huH2"] = dataset["huH2"].values / 100
        
        return dataset
    
    def to_float_test(self,dataset) :
        for i in dataset.iloc[:,1:29].columns:
            if type(dataset[i][0]) == str :
                dataset[i].replace(",",".",regex=True,inplace=True)
                dataset[i] = dataset[i].astype(np.float64)
        dataset["huH2"] = dataset["huH2"].values / 100
        return dataset
    
    def to_date(self,dataset):
        
        dataset['date'] = pd.to_datetime(dataset['date'])
        return dataset
    
    def to_dummy(self,dataset,colname) :
        # Attention au trap, on drop une dummy
        onehot = pd.get_dummies(dataset[colname], drop_first = True)
        onehot = onehot.rename(columns=lambda x: colname + '_' + str(x))
        dataset = dataset.join(onehot)
        
            
        return dataset
    
    # Dealing with missing data
    #l'objet groupby se compose de key et de l'item : key correspond a ce que l'on a groupé
    #alors que l'item est un sous dataframe issue de notre key.

    def missing_data_mean(self,dataset):
        colonne = dataset.iloc[:,1:43].columns.difference(["insee","mois"])
        for col in colonne :
            dataset[col] = dataset.iloc[:,1:43].groupby(['insee','mois']).transform(lambda x: x.fillna(x.mean()))[col]
        return dataset
        
            
    def drop(self,dataset,colname):
        
        dataset = dataset.drop(colname, axis = 1)
        
        return dataset
    
    def to_csv(self,dataset, filename):
        dataset.to_csv("./Datasets/"+ filename +'.csv',sep=';',index=False, encoding = "utf-8",decimal = ",")
        return dataset
    
    
class DataSet:
    def __init__(self,dataset):
        X = dataset[dataset.columns.difference(["tH2_obs" , "date"])]
        y = dataset["tH2_obs"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n = len(dataset)
        

    
# Preprocessing sur le train

d = pd.DataFrame()       
prp = Preprocessing()
dataset = prp.importing_data(d)
dataset = prp.to_float_train(dataset)
dataset = prp.to_date(dataset)
dataset = prp.to_dummy(dataset,"insee")
dataset = prp.to_dummy(dataset,"mois")
dataset = prp.missing_data_mean(dataset)
dataset = prp.drop(dataset,"mois")
dataset = prp.to_csv(dataset,"amira_t")

# Preprocessing sur le test
prp = Preprocessing()
dataset_t = pd.read_csv("./Datasets/test.csv", sep = ";", decimal = ","  , encoding = "utf-8")
dataset_t = prp.to_float_test(dataset_t)
dataset_t = prp.to_date(dataset_t)
dataset_t = prp.to_dummy(dataset_t,"insee")
dataset_t = prp.to_dummy(dataset_t,"mois")
dataset_t = prp.missing_data_mean(dataset_t)
dataset_t = prp.drop(dataset_t,"mois")
dataset_t["mois_janvier"] = 0
dataset_t["mois_février"] = 0
dataset_t["mois_mars"] = 0
dataset_t["mois_avril"] = 0
dataset_t = prp.to_csv(dataset_t,"amira_test")

# Scaling

sc_X = StandardScaler()
X_train = sc_X.fit_transform(d.X_train)
X_test = sc_X.transform(d.X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(d.y_train)




