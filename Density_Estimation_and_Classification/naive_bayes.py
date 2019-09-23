from scipy.stats import norm
import os
import numpy as np
import pandas as pd
import math

X = np.load('Data/train.npy')
y = np.load('Data/train_labels.npy')
X_test = np.load('Data/test.npy')
y_test = np.load('Data/test_labels.npy')

def compute_accuracy(y_true, y_pred):
    cm = np.zeros((2, 2)) # Confusion matrix
    corect = 0
    for true, pred in zip(y_test, y_pred):
        if true == pred:
            corect += 1
            if true==0:
                cm[0][0] += 1
            else:
                cm[1][1] +=1
        else:
            if true==0:
                cm[0][1] += 1
            else:
                cm[1][0] += 1
    # zip(y_test.reshape(2002,), y_pred.reshape(2002,)
    cm
    print "Accuracy: {}".format(corect/2002.)
    print "Confusion matrix (Diagonal is corectly classified)"
    print cm

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
print mu_0,
print "Mean values for class 8"
print mu_1
print "Stddev values for Class 7"
print np.sqrt(var_0)
print "Stddev values for Class 8"
print np.sqrt(var_1)
print
# Univariate gaussian function
def gaussian(x, mu, var):
    return 1./np.sqrt(2*math.pi*var) * np.exp(-np.power(x - mu, 2.) / (2 * var))

### Test with class 0
p_y_0 = float(inds_0.shape[0])/(inds_0.shape[0] + inds_1.shape[0]) #Prior prob of class 7
print "Prior of 7"
print p_y_0
x0_y0 = norm.pdf(X_test[:, 0], mu_0[0], math.sqrt(var_0[0])) # P(x0|y=0)
x1_y0 = norm.pdf(X_test[:, 1], mu_0[1], math.sqrt(var_0[1])) # P(x1|y=0)

#norm.pdf(X_test, mu_0[0], var_0[0])

### Test with class one
p_y_1 = float(inds_1.shape[0])/(inds_0.shape[0] + inds_1.shape[0])
print "Prior of 8"
print p_y_1
p_y_1
x0_y1 = norm.pdf(X_test[:, 0], mu_1[0], math.sqrt(var_1[0])) # P(x0|y=1)
x1_y1 = norm.pdf(X_test[:, 1], mu_1[1], math.sqrt(var_1[1])) # P(x1|y=1)

col_1 = (p_y_0 * x0_y0 * x1_y0).reshape(X_test.shape[0], 1) # Numerator class 7
col_2 = (p_y_1 * x0_y1 * x1_y1).reshape(X_test.shape[0], 1) # Numerator class 8

prob_mat = np.hstack((col_1, col_2)) # Prob pairs for each class
y_pred = np.argmax(prob_mat, axis=1) # Assigning max prob
compute_accuracy(y_test, y_pred) # Function to compute accuracy
