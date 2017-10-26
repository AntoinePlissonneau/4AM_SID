# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd
import sklearn.model_selection as s
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
    onehot = pd.get_dummies(dataset[colname], drop_first = True)
    onehot = onehot.rename(columns=lambda x: colname + '_' + str(x))
    dataset = dataset.join(onehot)
    return dataset


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

#a améliorer
def drop_na(dataset):
    dataset = dataset.dropna(subset = dataset.columns.difference(["rr1SOL0","flvis1SOL0","flsenSOL0","flir1SOL0","fllat1SOL0"]))
    return dataset

def na_predict(dataset) : 
    for i in dataset.columns:
        if dataset[i].isnull().sum() != 0 :
            dataset[i+ '_na'] = dataset[i].isnull().astype(int)
            dataset[i] = dataset[i].fillna(value=0)
    return dataset
            


class DataSet:
    def __init__(self,train,test = 'ok'):
        X = train[train.columns.difference(["tH2_obs","date"])]
        y = train["tH2_obs"]
        if type(test) != pd.pandas.core.frame.DataFrame:
            X_train, X_test, y_train, y_test = s.train_test_split(X, y, test_size = 0.2, random_state = 1)
            self.X_train = X_train 
            self.X_test = X_test 
            self.y_train = y_train 
            self.y_test = y_test 
            self.n = len(train)
        else:
            self.X_train = X
            self.y_train = y
            self.X_test = test[test.columns.difference(["date"])]


# Preprocessing sur le train

train = import_train()
train = to_float(train)
#train = to_date(train)
train = to_dummy(train,"insee")
train = to_dummy(train,"mois")
train = to_dummy(train,'ech')
train = drop_na(train)
train = na_predict(train)
#train = missing_data_mean(train)
train = drop(train,"mois")
train["flir1SOL0_na"] = 0
train["flsen1SOL0_na"] = 0
train["rr1SOL0_na"] = 0


to_csv(train,"PP_train")

# Preprocessing sur le test

test = pd.read_csv("./Datasets/test.csv", sep = ";", decimal = ","  , encoding = "utf-8")
test = to_float(test)
#test = to_date(test)
test = to_dummy(test,"insee")
test = to_dummy(test,"mois")
test = to_dummy(test,"ech")
test = na_predict(test)

#test = drop_na(test)
#test = missing_data_mean(test)
test = drop(test,"mois")
#test = drop(test,"ech")
#
test["mois_janvier"] = 0
test["mois_février"] = 0
test["mois_mars"] = 0
test["mois_avril"] = 0

test = test.loc[:,train.columns.difference(['tH2_obs'])]

to_csv(test,"PP_test")



test.isnull().sum().sum()
