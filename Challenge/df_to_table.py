# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 00:19:06 2017

@author: Amira AYADI
"""

#

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable


def series2descriptor(d):
    if d.dtype is np.dtype("float") or d.dtype is np.dtype("int"):
        return ContinuousVariable(str(d.name))
    else:
        t = d.unique()
        t.sort()
        return DiscreteVariable(str(d.name), list(t.astype("str")))

def df2domain(df):
    featurelist = [series2descriptor(df.iloc[:,col]) for col in range(len(df.columns))]
    return Domain(featurelist)

def df2table(df):
    tdomain = df2domain(df)
    ttables = [series2table(df.iloc[:,i], tdomain[i]) for i in range(len(df.columns))]
    ttables = np.array(ttables).reshape((len(df.columns),-1)).transpose()
    return Table(tdomain , ttables)

def series2table(series, variable):
    if series.dtype is np.dtype("int") or series.dtype is np.dtype("float"):
        series = series.values[:, np.newaxis]
        return Table(series)
    else:
        series = series.astype('category').cat.codes.reshape((-1,1))
        return Table(series)

ot = df2table(dataset)
imputer = Orange.feature.imputation.ModelConstructor()
imputer.learner_continuous = imputer.learner_discrete = Orange.classification.tree.TreeLearner(min_subset=20)
imputer = imputer(ot)