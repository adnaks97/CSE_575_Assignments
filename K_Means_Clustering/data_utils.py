import scipy.io
import numpy as np

NumpyFile = scipy.io.loadmat('Data/AllSamples.mat')
X = NumpyFile['AllSamples']

X.shape
np.save('Data/data.npy', X)
