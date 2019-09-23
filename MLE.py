from scipy.stats import norm
import os
import numpy as np
import pandas as pd
import math

X = np.load('Data/train.npy')
y = np.load('Data/train_labels.npy')
X_test = np.load('Data/test.npy')
y_test = np.load('Data/test_labels.npy')

'''
MLE PARAMETER ESTIMATION
'''
inds_0 = np.where(y==0)[0] # Indices of 7
inds_1 = np.where(y==1)[0] # Indices of 8

mu_0 = np.mean(X[inds_0], axis=0)
mu_1 = np.mean(X[inds_1], axis=0)

var_0 = np.var(X[inds_0], axis=0)
var_1 = np.var(X[inds_1], axis=0)

print "Mean values for Class 7"
print mu_0
print "Mean values for class 8"
print mu_1
print "Stddev values for Class 7"
print np.sqrt(var_0)
print "Stddev values for Class 8"
print np.sqrt(var_1)
print
