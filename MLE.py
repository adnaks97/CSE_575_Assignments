import os
import numpy as np
import pandas as pd
import math
import sklearn

X = np.load('Data/train.npy')
y = np.load('Data/train_labels.npy')

y.shape
X.shape

inds_7 = np.where(y==0)[0] # Indices of 7
inds_8 = np.where(y==1)[0] # Indices of 8

u1 = np.mean(X[inds_7], axis=0)
u1
u2 = np.mean(X[inds_8], axis=0)
u2

var1 = np.var(X[inds_7], axis=0)
var2 = np.var(X[inds_8], axis=0)
var1
var2

sum=np.zeros((1,2))
for sample in X[inds_7]:
    diff = sample - u2
    sum += diff*diff.T
    #print sum
sum/X[inds_8].shape[0]
sum

from sklearn.covariance import EmpiricalCovariance
cov = EmpiricalCovariance().fit(X[inds_8])
cov
cov.covariance_

(X[inds_7][0] - u1)*(X[inds_7][0] - u1).T
#np.matmul((X[inds_7][0] - u1),(X[inds_7][0] - u1).T)
np.matmul(np.array([2, 2]), np.array([3, 3]))
np.dot(np.array([2, 2]), np.array([3, 3]))
(np.array([2, 2]) * np.array([3, 3]))

X[inds_7]
np.unique(X[inds_7][:,0]).shape
inds_7.shape
np.argmax([0.1, 0.4, 0.3, 0.6])
from scipy.stats import multivariate_normal
X[3], y[3]
mvn.logpdf(X[6525], u1, np.sqrt(var1))
y[6525]

cov_matrix = np.array([[np.var(X[inds_7, 1]), 0], [0, np.var(X[inds_7])]])
cov_matrix
mvn = multivariate_normal(mean=u1, cov=cov_matrix)
mvn.pdf(X[inds_7])
inds_7
np.var(X[inds_8, 1])
np.var(X[inds_8, 0])

mvn.pdf(3, 3, 1.58)
np.sqrt(var1)
var1

from scipy.stats import norm
norm.pdf(1.1, 20, 5)
