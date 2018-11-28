

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



def get_normalized_data():
    print("Reading in and transforming data...")


    df = pd.read_csv('train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:]
    Ytest  = Y[-1000:]

    # normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    
    return Xtrain, Xtest, Ytrain, Ytest




def softmax_forward(X, W, b):
    # softmax
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y


def predict(p_y):
    return np.argmax(p_y, axis=1)


def cal_cal_error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)


def calculate_cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()


def gradW(t, y, X):
    return X.T.dot(t - y)


def gradb(t, y):
    return (t - y).sum(axis=0)


def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def Fun_GradientDecent():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    print("Performing logistic regression...")
    # lr = LogisticRegression(solver='lbfgs')


    # convert Ytrain and Ytest to (N x K) matrices of indicator variables
    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    lr = 0.00004 
    reg = 0.01

    for i in range(500):
        p_y = softmax_forward(Xtrain, W, b)
        # print "p_y:", p_y
        ll = calculate_cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = softmax_forward(Xtest, W, b)
        lltest = calculate_cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)
        
        err = cal_cal_error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("calculate_cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = softmax_forward(Xtest, W, b)
    print("Final error rate:", cal_cal_error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()



if __name__ == '__main__':
    Fun_GradientDecent()

