import pandas as pd
import random
import numpy as np

def readin(filename):
    fo = open('逻辑回归/' + filename, 'r')
    X = []
    Label = []
    for line in fo.readlines():
        line = line.strip().split(',')
        X.append([1, float(line[0]), float(line[1])])
        Label.append([int(line[2])])
    X = np.mat(X)
    Label = np.mat(Label)
    fo.close()
    return X, Label

def Logic_Regression(X, Label, alpha, times):
    x1_min = np.min(X[...,1]); x1_max = np.max(X[...,1])
    x2_min = np.min(X[...,2]); x2_max = np.max(X[...,2])

    w2 = 1
    w1 = (x2_max - x2_min) / (x1_max - x1_min)
    w0 = -x2_max - w1 * x1_min
    W = [w0, w1, w2]
    W = np.mat(W).T
    for i in range(times):
        H = 1 / (np.exp(-X * W) + 1)
        dC_by_dW = X.T * (H - Label)
        W -= alpha * dC_by_dW
    return W

def Draw(X, Label, k = 0, b = 0):
    import matplotlib.pyplot as plt
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    for i in range(X.shape[0]):
        if Label[i] == 0:
            X1.append(X[i, 1])
            X2.append(X[i, 2])
        else:
            X3.append(X[i, 1])
            X4.append(X[i, 2])
    point_0 = [np.array(X1), np.array(X2)]
    point_1 = [np.array(X3), np.array(X4)]
    plt.scatter(point_0[0], point_0[1], c="red")
    plt.scatter(point_1[0], point_1[1], c="blue")
    nums = np.arange(np.min(X[...,1])-1, np.max(X[...,1])+1)
    plt.plot(nums, k * nums + b)
    plt.show()


X, Label = readin('data.txt')
W = Logic_Regression(X, Label, 0.01, 10000).T
Draw(X, Label, (-W[0, 1] / W[0, 2]), (-W[0, 0] / W[0, 2]))