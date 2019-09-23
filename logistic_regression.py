from scipy.stats import norm
import os
import numpy as np
import pandas as pd
import math
import sklearn

NUM_FEATURES = 2
lr = 0.01

X = np.load('Data/train.npy')
y = np.load('Data/train_labels.npy')
X_test = np.load('Data/test.npy')
y_test = np.load('Data/test_labels.npy')
print(X.shape, y.shape, X_test.shape, y_test.shape)

weights = np.random.random(size=(1, 3))
X = np.append(X, np.ones((X.shape[0], 1), dtype=float), axis=1)
X_test = np.append(X_test, np.ones((X_test.shape[0], 1), dtype=float), axis=1)

n_train = X.shape[0] # N_train samples
n_test = X_test.shape[0]

def compute_accuracy(y_true, y_pred):
    cm = np.zeros((2, 2))
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
    print "Confusion matrix"
    print cm

def sigmoid(z):
    '''
    @params z: float array
    '''
    return 1./(1 + np.exp(-z))

inds_0 = np.where(y==0)[0] # Indices of 7
inds_1 = np.where(y==1)[0] # Indices of 8

def calc_grads(X, y, yhat, batch_size):
    #grads = (y-yhat)*X
    error = y-yhat
    grads = np.dot(error.T, X) # delta= y^T.X
    grads /= batch_size # Averaging grads
    return grads

def weight_update(X, y, yhat, W, lr=0.001, batch_size=n_train):
    i=0
    while i<n_train:
        start = i
        if i+batch_size <= n_train:
            stop = i+batch_size
        else:
            stop = n_train
        gradients = calc_grads(X[start:stop], y[start:stop], yhat[start:stop], stop-start)
        #print "grads: {}".format(gradients.shape)
        W = W + lr*gradients
        i += batch_size
    return W

def predict(X, W):
    Z = np.sum(X*W, axis=1)
    yhat = sigmoid(Z).reshape(X.shape[0], 1)
    return yhat

def calc_cost(y, yhat):
    cost = y*np.log(yhat) + (1-y)*np.log(1-yhat)
    cost = cost.sum() / n_train
    return cost

def train(X, y, W, lr=0.001, n_iters=1000, batch_size=n_train):
    '''
    X: train Data
    y: test labels
    lr: learning rate
    n_iters: number of iterations
    batch_size: number of samples to cimpute gradiets at a time
    '''
    cost_log = []
    tol = 1e-4
    i=0; delta=100
    for i in range(n_iters):
        yhat = predict(X, W)
        cost = calc_cost(y, yhat)
        cost_log.append(cost)
        W = weight_update(X, y, yhat, W, lr, batch_size)
        if i%100 == 0:
            print ("Cost at iter {} is {}".format(i, cost_log[-1]))
        i += 1
        if i>1:
            delta = cost_log[-1]- cost_log[-2]
    return W, cost_log, yhat

def bound(prob):
    return 1 if prob > 0.5 else 0

if __name__ == '__main__':
    print weights
    W, cost_log, yhat = train(X, y, weights, lr=0.5, n_iters=40000, batch_size=n_train)
    print "Weights"
    print W
    yhat = predict(X_test, W)
    print "Cost of test: "
    print calc_cost(y_test, yhat)
    make_prob = np.vectorize(bound) # Vectorizing the function
    y_pred = make_prob(yhat) # Bound probs to zero and 1
    compute_accuracy(y_test, y_pred)

    # Plot iteration vs cost plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.plot(range(40000), cost_log)
    plt.title("Iterations vs Cost")
    plt.xlabel("n_iterations")
    plt.ylabel("cost")
    plt.savefig('cost_convergence.png')
    plt.show()
