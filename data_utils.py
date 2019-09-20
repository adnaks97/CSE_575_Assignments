import numpy as np
import os
import pandas as pd
import scipy.io
Numpyfile = scipy.io.loadmat('Data/mnist_data.mat')
Numpyfile
trX, trY, tsX, tsY = Numpyfile['trX'], Numpyfile['trY'], Numpyfile['tsX'], Numpyfile['tsY']
print trX.shape
print tsX.shape

#np.var(trX, axis=1)
#np.mean(trX, axis=1).shape
def extract_features(arr):
    mean = np.mean(arr, axis=1).reshape(arr.shape[0], 1)
    var = np.var(arr, axis=1).reshape(arr.shape[0], 1)
    feat = np.hstack((mean, var))
    return feat

X, X_test = extract_features(trX), extract_features(tsX)
y = trY.reshape(trY.shape[1], 1)
y_test = tsY.reshape(tsY.shape[1], 1)
#np.mean(trX, axis=1)
#np.var(trX, axis=1)
np.save('Data/train.npy', X)
np.save('Data/train_labels.npy', y)
np.save('Data/test.npy', X_test)
np.save('Data/test_labels.npy', y_test)
