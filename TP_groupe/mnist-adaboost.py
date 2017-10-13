# -*- coding: utf-8 -*-
"""
Created on Tue Oct 9 16:15:47 2017

@author: Amira AYADI
"""

import time
from pylab import * 
from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

#data
mnist = fetch_mldata('MNIST original', data_home='./')
mnist.target = mnist.target.astype(int) # by default the digits are floating numbers: convert to integers

# defining general variables for use throuhout the notebook  
n_tot = len(mnist.data)

class DataSet:
    def __init__(self, data, target): 
        self.data = data # input : images
        self.X = data # default: features = data itself
        self.target = target # output: labels
        self.n = len(data)

def evaluate(classifier, X, y):
    return(mean(classifier.predict(X) != y))

# training set

n_train = 10000
I = random.choice(range(n_tot), n_train, replace=False)
trainingSet = DataSet(mnist.data[I, :], mnist.target[I]) 

# testing set

mask = ones(n_tot, dtype=bool)
mask[I] = False
testingSet = DataSet(mnist.data[mask, :], mnist.target[mask])

#set a classifier
deci_tree = DecisionTreeClassifier(max_depth=10, criterion="entropy") 

#test de plueisurs méthodes en changeant plusieurs parametres :
    
learning_rate = [1,0.7,0.5,0.1]
algorythm = ["SAMME","SAMME.R"]
n_estimator = [50,100,200,500]
color = ["red","blue","green","orange"]

#initialisation figure

fig = plt.figure()
ax = fig.add_subplot(111)

#début test pour chacun des deux algo, on modifie le learning rate et le n_estmator:

for algo in algorythm:
    for i in range(len(learning_rate)):
        start_time = time.time()
        model = AdaBoostClassifier(
                base_estimator = deci_tree,
                algorithm = algo,
                n_estimators = n_estimator[i],
                learning_rate = learning_rate[i] )
        model.fit(trainingSet.data,trainingSet.target)
        print ('le temps d\'execution avec {0},{1},{2} en secondes est  : {3} '.format(algo,str(learning_rate[i]),str(n_estimator[i]), time.time() - start_time))
        print("Proportion of misclassified images in training set: %g" %(evaluate(model, trainingSet.X, trainingSet.target)))
        print("Proportion of misclassified images in testing set: %g" %(evaluate(model, testingSet.X, testingSet.target)))

        ax.plot([evaluate(model, testingSet.X, testingSet.target).item()],
                [n_estimator[i]],
                label=algo+'-'+str(learning_rate[i]),
                color=color[i],
                marker='o')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error')
    
    leg = ax.legend(loc='upper right', fancybox=True)
    
    plt.show()

#meilleur résultat avec 500, 0.7 et samme l'erreur est de 3,8%
        
#essai avec fonction zero one loss + meilleur graph !



#initialisation figure

fig = plt.figure()
ax = fig.add_subplot(111)

#début test pour chacun des deux algo, on modifie le learning rate et le n_estmator:

for algo in algorythm:
    for i in range(len(learning_rate)):
        
        start_time = time.time()
        
        model = AdaBoostClassifier(
                base_estimator = deci_tree,
                algorithm = algo,
                n_estimators = n_estimator[i],
                learning_rate = learning_rate[i] )
        
        model.fit(trainingSet.data,trainingSet.target)
        
        print ('le temps d\'execution avec {0},{1},{2} en secondes est  : {3} '.format(algo,str(learning_rate[i]),str(n_estimator[i]), time.time() - start_time))
        
        a = zeros((n_estimator[i],))
        
        for y, y_pred in enumerate(model.staged_predict(testingSet.X)):
            a[y] = zero_one_loss(y_pred, testingSet.target)

        ax.plot(arange(n_estimator[i])+1 ,
                a,
                label=algo+'-'+str(learning_rate[i]),
                color=color[i],
                marker='o')
    ax.plot(arange(n_estimator[i])+1 ,
                a,
                label=algo+'-'+str(learning_rate[i]),
                color=color[i],
                marker='o')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error')
    
    leg = ax.legend(loc='upper right', fancybox=True)
    
    plt.show()
    

