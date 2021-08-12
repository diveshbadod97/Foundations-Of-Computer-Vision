import numpy as np
import pandas as pd
import csv


def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def init_weight_and_bias_NN(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    return -(T*np.log(Y)).sum()
    
    
def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def getBinaryfer13Data(filename):
    Y = []
    X = []
    first = True
    for line in open(filename, 'rU'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            type = row[2]
            if y == 3 or y == 4:	#3=happy; 4=sad
                Y.append(abs(y-4))  #we want to store 1 for happy, 0 for sad
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


def ReLU(x):
    return x * (x > 0)
