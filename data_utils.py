import numpy as np
import os
import pandas as pd
import scipy.io
Numpyfile = scipy.io.loadmat('Data/mnist_data.mat')
Numpyfile
trX, trY, tsX, tsY = Numpyfile['trX'], Numpyfile['trY'], Numpyfile['tsX'], Numpyfile['tsY']
print trX.shape
print tsX.shape

def extract_features(arr):
    # Calculating mean and stddev for each image
    mean = np.mean(arr, axis=1).reshape(arr.shape[0], 1)
    var = np.std(arr, axis=1).reshape(arr.shape[0], 1)
    feat = np.hstack((mean, var))
    return feat

X, X_test = extract_features(trX), extract_features(tsX)
y = trY.reshape(trY.shape[1], 1)
y_test = tsY.reshape(tsY.shape[1], 1)

# Feature extracted data
np.save('Data/train.npy', X)
np.save('Data/train_labels.npy', y)
np.save('Data/test.npy', X_test)
np.save('Data/test_labels.npy', y_test)

# Visualize data distribution
import matplotlib.pyplot as plt
inds_0 = np.where(y==0)[0] # Indices of 7
inds_1 = np.where(y==1)[0] # Indices of 8
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[inds_0, 0], X[inds_0, 1], alpha=0.8, c='red', edgecolors='none', s=30, label='7')
ax.scatter(X[inds_1, 0], X[inds_1, 1], alpha=0.8, c='blue', edgecolors='none', s=30, label='8')
plt.title('Distribution of data')
plt.xlabel('Mean')
plt.ylabel('Stddev')
plt.legend(loc=2)
plt.savefig('Data_distribution.png')
plt.show()
