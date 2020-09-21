import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import *
import random
import time


def q1(x1, x2, y, X1, X2, Y):
    print("BATCH GRADIENT DESCENT")
    print(" ")
    w0, w1, w2, cost, w1_list, w2_list = batch_gradient_descent(x1, x2, y)
    print("W0 :", w0)
    print("W1 :", w1)
    print("W2 :", w2)
    print(" ")
    plot2d("cost", "iterations", cost, "Batch-Gradient-Descent")
    print(" ")
    plot3d(cost, w1_list, w2_list, "cost", "w1", "w2", "Batch-Gradient-Descent")
    print(" ")
    pred(w0, w1, w2, X1, X2, Y)
    time.sleep(5)


def q2(x1, x2, y, X1, X2, Y):
    print(" ")
    print(" ")
    print(" ")
    print("MINI BATCH GRADIENT DESCENT")
    print(" ")
    w0, w1, w2, cost, w1_list, w2_list = mini_batch(x1, x2, y)
    print("W0 :", w0)
    print("W1 :", w1)
    print("W2 :", w2)
    print(" ")
    plot2d("cost", "iterations", cost, "Mini-Batch-Gradient-Descent")
    print(" ")
    plot3d(cost, w1_list, w2_list, "cost", "w1", "w2", "Mini-Batch-Gradient-Descent")
    print(" ")
    pred(w0, w1, w2, X1, X2, Y)
    print(" ")
    print(" ")
    print(" ")
    time.sleep(5)
    print("STOCHASTIC GRADIENT DESCENT")
    w0, w1, w2, cost, w1_list, w2_list = stochastic_gradient_descent(x1, x2, y)
    print("W0 :", w0)
    print("W1 :", w1)
    print("W2 :", w2)
    print(" ")
    plot2d("cost", "iterations", cost, "Stochastic-Gradient-Descent")
    print(" ")
    plot3d(cost, w1_list, w2_list, "cost", "w1", "w2", "Stochastic-Gradient-Descent")
    print(" ")
    pred(w0, w1, w2, X1, X2, Y)
    time.sleep(5)


def minmaxer(X_train, Y_train, X_test, Y_test):
    X_train = MinMax(X_train)
    Y_train = MinMax(Y_train)
    x1, x2, y = split(X_train, Y_train)
    w0, w1, w2, cost = batch_gradient_descent(x1, x2, y)
    print(w0)
    print(w1)
    print(w2)
    plot2d("cost", "iterations", cost)


def meansdever(X_train, Y_train, X_test, Y_test):
    X_train = Normalize(X_train)
    Y_train = Normalize(Y_train)
    X_test = Normalize(X_test)
    Y_test = Normalize(Y_test)
    x1, x2, y = split(X_train, Y_train)
    X1, X2, Y = split(X_test, Y_test)
    q1(x1, x2, y, X1, X2, Y)
    q2(x1, x2, y, X1, X2, Y)


def main():
    X_train = read('training_feature_matrix')
    Y_train = read('training_output')
    X_test = read('test_feature_matrix')
    Y_test = read('test_output')
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    '''
    rows = dataframe.shape[0]
    columns = dataframe.shape[1]
    '''
    # minmaxer(X_train, Y_train, X_test, Y_test)
    meansdever(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    main()