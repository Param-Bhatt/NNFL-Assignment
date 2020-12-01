import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import *
from functions_v import *
import random
import time
from Q1 import *
from Q3 import *
from Q4 import *
from Q5 import *
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
    #q1(x1, x2, y, X1, X2, Y)
    #q2(x1, x2, y, X1, X2, Y)
    #q3(x1, x2, y, X1, X2, Y)
    #q4(x1, x2, y, X1, X2, Y)
    q5(x1, x2, y, X1, X2, Y)


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