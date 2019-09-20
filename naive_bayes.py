from scipy.stats import norm
import os
import numpy as np
import pandas as pd
import math
import sklearn

X = np.load('Data/train.npy')
y = np.load('Data/train_labels.npy')
X_test = np.load('Data/test.npy')
y_test = np.load('Data/test_labels.npy')

inds_0 = np.where(y==0)[0] # Indices of 7
inds_1 = np.where(y==1)[0] # Indices of 8

mu_0 = np.mean(X[inds_0], axis=0)
mu_1 = np.mean(X[inds_1], axis=0)

var_0 = np.var(X[inds_0], axis=0)
var_1 = np.var(X[inds_1], axis=0)

mu_0, mu_1
var_0, var_1

def gaussian(x, mu, var):
    return 1./np.sqrt(2*math.pi*var) * np.exp(-np.power(x - mu, 2.) / (2 * var))

### Test with class 0
p_y_0 = float(inds_0.shape[0])/(inds_0.shape[0] + inds_1.shape[0])
p_y_0
X[inds_0][0][1], mu_0[0], var_0[0]
x0_y0 = norm.pdf(X_test[:, 0], mu_0[0], math.sqrt(var_0[0])) # P(x0|y=0)
x1_y0 = norm.pdf(X_test[:, 1], mu_0[1], math.sqrt(var_0[1])) # P(x1|y=0)

#norm.pdf(X_test, mu_0[0], var_0[0])

### Test with class one
p_y_1 = float(inds_1.shape[0])/(inds_0.shape[0] + inds_1.shape[0])
p_y_1
x0_y1 = norm.pdf(X_test[:, 0], mu_1[0], math.sqrt(var_1[0])) # P(x0|y=1)
x1_y1 = norm.pdf(X_test[:, 1], mu_1[1], math.sqrt(var_1[1])) # P(x1|y=1)

col_1 = (p_y_0 * x0_y0 * x1_y0).reshape(X_test.shape[0], 1)
col_2 = (p_y_1 * x0_y1 * x1_y1).reshape(X_test.shape[0], 1)

prob_mat = np.hstack((col_1, col_2))
y_pred = np.argmax(prob_mat, axis=1)
col_1.shape
col_2.shape
prob_mat.shape

from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, np.argmax(prob_mat, axis=1))
accuracy_score(y_test, y_pred)

from sklearn.naive_bayes import GaussianNB, MultinomialNB
gnb = GaussianNB()
mnb = MultinomialNB()
y_pred_sklearn = gnb.fit(X, y).predict(X_test)
y_pred_sklearn_mnb = mnb.fit(X, y).predict(X_test)
accuracy_score(y_test, y_pred_sklearn)

norm.pdf(X_test[0][0], mu_0[0], math.sqrt(var_0[0]))
gaussian(X_test[0][0], mu_0[0], var_0[0])

var_0[0]
np.sqrt(var_0[0])
