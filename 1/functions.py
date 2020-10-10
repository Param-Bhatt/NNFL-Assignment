import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def read(name):
    return pd.read_csv(name + '.csv', header=None)


def getcol(data, c):
    col = []
    var = data.shape[0]
    for i in range(0, data.shape[0]):
        col.append(data[i][c])
    return col


def split(data, op):
    # specific to question 1. add cases accordingly
    x1 = data[0:, 0]
    x2 = data[0:, 1]
    y = op[0:, 0]
    return x1, x2, y


def split1(data):
    x1 = data[0:, 0]
    x2 = data[0:, 1]
    y = data[:, 2]
    return x1, x2, y


def MinMax(data):
    temp = data
    for i in range(0, 1):
        maxval = max(getcol(data, i))
        minval = min(getcol(data, i))
        for j in range(0, data.shape[0]):
            temp[j][i] = (data[j][i] - minval) / (maxval - minval)
    return temp


def Normalize(data):
    mean = np.ones(data.shape[1])
    sdev = np.ones(data.shape[1])
    for i in range(0, data.shape[1]):
        mean[i] = np.mean(data.transpose()[i])
        sdev[i] = np.std(data.transpose()[i])
        for j in range(0, data.shape[0]):
            data[j][i] = (data[j][i] - mean[i]) / sdev[i]
    return data


def hypothesis(w0, w1, w2, x1, x2):
    hyp = np.ones(x1.shape[0])
    for i in range(0, x1.shape[0]):
        hyp[i] = w0 + w1 * x1[i] + w2 * x2[i]

    return hyp


def summa(hyp, rows, y_data, x_data):
    sum = 0
    for i in range(0, rows):
        sum = sum + (hyp[i] - y_data[i]) * x_data[i]
    return sum


def mse(h, y, rows):
    sum = 0
    for i in range(0, rows):
        sum = sum + (h[i] - y[i]) ** 2
    return sum

def wsquare(w0,w1,w2):
    sum = 0
    sum = sum + w0 ** 2
    sum = sum + w1 ** 2
    sum = sum + w2 ** 2
    return sum

def wabs(w0,w1,w2):
  w0 = abs(w0)
  w1 = abs(w1)
  w2 = abs(w2)
  return w0+w1+w2

def plot2d(label1, label2, y, title):
    fig = plt.figure()
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.plot(y)
    plt.title(title + " : "+label1 + "  vs  " + label2)
    plt.xlabel(label2)
    plt.ylabel(label1)
    plt.show()


def plot3d(x, y, z, label1, label2, label3, title):
    fig = plt.figure()
    plt.style.use('dark_background')
    plt.title(title + " : " + label1 + "  vs  " + label2 + " vs "+ label3)
    plt.rcParams["figure.figsize"] = (10, 10)
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'red')
    ax.set_xlabel(label1, fontsize=18)
    ax.set_ylabel(label2, fontsize=18)
    ax.set_zlabel(label3, fontsize=18)


def pred(w0, w1, w2, X1, X2, Y):
    pred = hypothesis(w0, w1, w2, X1, X2)
    mean = np.mean(pred.transpose())
    sdev = np.std(pred.transpose())
    # print(mean)
    # print(sdev)
    pred = pred * sdev + mean
    err = 0
    for i in range(0, pred.shape[0]):
        err = err + (pred[i] - Y[i]) ** 2
    err = err / pred.shape[0]
    print("MSE :", err)


def batch_gradient_descent(x1, x2, y):
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    x0 = np.ones(x1.shape[0])
    iters = 300
    alpha = 0.0001
    h = hypothesis(w0, w1, w2, x1, x2)
    cost = [0 for i in range(iters)]
    w0_list = [1 for i in range(iters)]
    w1_list = [1 for i in range(iters)]
    w2_list = [1 for i in range(iters)]
    for i in range(0, iters):
        w0 = w0 - alpha * summa(h, y.shape[0], y, x0)
        w1 = w1 - alpha * summa(h, y.shape[0], y, x1)
        w2 = w2 - alpha * summa(h, y.shape[0], y, x2)
        w0_list[i] = w0
        w1_list[i] = w1
        w2_list[i] = w2
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[i] = 1 * 0.5 * mse(h, y, y.shape[0])
    return w0, w1, w2, cost, w1_list, w2_list

def ridge_batch_gradient_descent(x1, x2, y):
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    x0 = np.ones(x1.shape[0])
    iters = 300
    alpha = 0.0001
    lamb = 0.05
    h = hypothesis(w0, w1, w2, x1, x2)
    cost = [0 for i in range(iters)]
    w0_list = [1 for i in range(iters)]
    w1_list = [1 for i in range(iters)]
    w2_list = [1 for i in range(iters)]
    for i in range(0, iters):
        w0 = (1-(alpha*lamb))*w0 - alpha * summa(h, y.shape[0], y, x0)
        w1 = (1-(alpha*lamb))*w1 - alpha * summa(h, y.shape[0], y, x1)
        w2 = (1-(alpha*lamb))*w2 - alpha * summa(h, y.shape[0], y, x2)
        w0_list[i] = w0
        w1_list[i] = w1
        w2_list[i] = w2
        wsum = wsquare(w0,w1,w2)
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[i] = (1 * 0.5 * mse(h, y, y.shape[0])/x1.shape[0] + lamb * wsum) * 0.5
    return w0, w1, w2, cost, w1_list, w2_list


def least_angle_regression_batch(x1,x2,y):
  lamb = 0.2
  alpha = 0.0005
  iter = 500
  # initializing the learning rates to random values between 0 & 1
  w0 = random.uniform(0, 1)
  w1 = random.uniform(0, 1)
  w2 = random.uniform(0, 1)
  # grabbing x0 as np.ones for future use
  x0 = np.ones(x1.shape[0])
  cost = [0 for i in range(iter)]
  w0_list = [1 for i in range(iter)]
  w1_list = [1 for i in range(iter)]
  w2_list = [1 for i in range(iter)]
  for i in range(0,iter):
    h = hypothesis(w0, w1, w2, x1, x2)
    w0 = w0 - alpha * summa(h, y.shape[0], y, x0) - alpha * lamb * np.sign(w0) * 0.5
    w1 = w1 - alpha * summa(h, y.shape[0], y, x1) - alpha * lamb * np.sign(w1) * 0.5
    w2 = w2 - alpha * summa(h, y.shape[0], y, x2) - alpha * lamb * np.sign(w2) * 0.5
    w0_list[i] = w0
    w1_list[i] = w1
    w2_list[i] = w2
    wsum = wabs(w0,w1,w2)
    cost[i] = (1 * 0.5 * mse(h, y, y.shape[0])/x1.shape[0] + lamb * wsum) * 0.5
  return w0, w1, w2, cost, w1_list, w2_list


def mini_batch(x1, x2, y):
    iters = 500
    batch = 5
    alpha = 0.00001
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    x0 = np.ones(x1.shape[0])
    # h = hypothesis(w0, w1, w2, x1, x2)
    cost = [0 for i in range(iters)]
    w0_list = [1 for i in range(iters)]
    w1_list = [1 for i in range(iters)]
    w2_list = [1 for i in range(iters)]
    for i in range(0, iters):
        temp = np.column_stack((x1, x2, y))
        np.random.shuffle(temp)
        x1, x2, y = split1(temp)
        # print(temp[:,2].shape)
        # print(x1.shape, " ", x2.shape, " ", y.shape)
        for j in range(0, batch):
            h = hypothesis(w0, w1, w2, x1, x2)
            w0 = w0 - alpha * summa(h, x0.shape[0], y, x0)
            w1 = w1 - alpha * summa(h, x1.shape[0], y, x1)
            w2 = w2 - alpha * summa(h, x2.shape[0], y, x2)
            w0_list[i] = w0
            w1_list[i] = w1
            w2_list[i] = w2
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[i] = 0.5 * mse(h, y, y.shape[0])
    return w0, w1, w2, cost, w1_list, w2_list

def ridge_mini_batch(x1, x2, y):
    iters = 500
    batch = 5
    alpha = 0.00001
    lamb = 0.05
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    x0 = np.ones(x1.shape[0])
    # h = hypothesis(w0, w1, w2, x1, x2)
    cost = [0 for i in range(iters)]
    w0_list = [1 for i in range(iters)]
    w1_list = [1 for i in range(iters)]
    w2_list = [1 for i in range(iters)]
    for i in range(0, iters):
        temp = np.column_stack((x1, x2, y))
        np.random.shuffle(temp)
        x1, x2, y = split1(temp)
        # print(temp[:,2].shape)
        # print(x1.shape, " ", x2.shape, " ", y.shape)
        for j in range(0, batch):
            h = hypothesis(w0, w1, w2, x1, x2)
            w0 = (1-(alpha*lamb))*w0 - alpha * summa(h, x0.shape[0], y, x0)
            w1 = (1-(alpha*lamb))*w1 - alpha * summa(h, x1.shape[0], y, x1)
            w2 = (1-(alpha*lamb))*w2 - alpha * summa(h, x2.shape[0], y, x2)
            w0_list[i] = w0
            w1_list[i] = w1
            w2_list[i] = w2
            wsum = wsquare(w0,w1,w2)
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[i] = (1 * 0.5 * mse(h, y, y.shape[0])/x1.shape[0] + lamb * wsum) * 0.5
    return w0, w1, w2, cost, w1_list, w2_list

def least_angle_mini_batch(x1, x2, y):
    iters = 500
    batch = 5
    alpha = 0.00001
    lamb = 0.05
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    x0 = np.ones(x1.shape[0])
    # h = hypothesis(w0, w1, w2, x1, x2)
    cost = [0 for i in range(iters)]
    w0_list = [1 for i in range(iters)]
    w1_list = [1 for i in range(iters)]
    w2_list = [1 for i in range(iters)]
    for i in range(0, iters):
        temp = np.column_stack((x1, x2, y))
        np.random.shuffle(temp)
        x1, x2, y = split1(temp)
        # print(temp[:,2].shape)
        # print(x1.shape, " ", x2.shape, " ", y.shape)
        for j in range(0, batch):
            h = hypothesis(w0, w1, w2, x1, x2)
            w0 = w0 - alpha * summa(h, y.shape[0], y, x0) - alpha * lamb * np.sign(w0) * 0.5
            w1 = w1 - alpha * summa(h, y.shape[0], y, x1) - alpha * lamb * np.sign(w1) * 0.5
            w2 = w2 - alpha * summa(h, y.shape[0], y, x2) - alpha * lamb * np.sign(w2) * 0.5
            w0_list[i] = w0
            w1_list[i] = w1
            w2_list[i] = w2
            wsum = wsquare(w0,w1,w2)
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[i] = (1 * 0.5 * mse(h, y, y.shape[0])/x1.shape[0] + lamb * wsum) * 0.5
    return w0, w1, w2, cost, w1_list, w2_list

def stochastic_gradient_descent(x1, x2, y):
    # batch = 15 #batch size
    alpha = 0.00001  # learning rate
    iter = 9999
    # initializing the learning rates to random values between 0 & 1
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    # grabbing x0 as np.ones for future use
    x0 = np.ones(x1.shape[0])
    cost = [0 for i in range(iter)]
    w0_list = [1 for i in range(iter)]
    w1_list = [1 for i in range(iter)]
    w2_list = [1 for i in range(iter)]
    h = hypothesis(w0, w1, w2, x1, x2)
    for itr in range(0, iter):
        i = random.randint(0, y.shape[0] - 1)
        w0 = w0 - alpha * ((h[i] - y[i]) * x0[i])
        w1 = w1 - alpha * ((h[i] - y[i]) * x1[i])
        w2 = w2 - alpha * ((h[i] - y[i]) * x2[i])
        w0_list[itr] = w0
        w1_list[itr] = w1
        w2_list[itr] = w2
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[itr] = 0.5 * mse(h, y, y.shape[0])
    return w0, w1, w2, cost, w1_list, w2_list

def ridge_stochastic_gradient_descent(x1, x2, y):
    # batch = 15 #batch size
    alpha = 0.00001  # learning rate
    lamb = 0.6
    iter = 9999
    # initializing the learning rates to random values between 0 & 1
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    # grabbing x0 as np.ones for future use
    x0 = np.ones(x1.shape[0])
    cost = [0 for i in range(iter)]
    w0_list = [1 for i in range(iter)]
    w1_list = [1 for i in range(iter)]
    w2_list = [1 for i in range(iter)]
    h = hypothesis(w0, w1, w2, x1, x2)
    for itr in range(0, iter):
        i = random.randint(0, y.shape[0] - 1)
        w0 = (1- (alpha* lamb ))* w0 - alpha * ((h[i] - y[i]) * x0[i])
        w1 = (1- (alpha* lamb ))* w1 - alpha * ((h[i] - y[i]) * x1[i])
        w2 = (1- (alpha* lamb ))* w2 - alpha * ((h[i] - y[i]) * x2[i])
        w0_list[itr] = w0
        w1_list[itr] = w1
        w2_list[itr] = w2
        wsum = wsquare(w0,w1,w2)
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[itr] = (1 * 0.5 * mse(h, y, y.shape[0])/x1.shape[0] + lamb * wsum) * 0.5
    return w0, w1, w2, cost, w1_list, w2_list

def least_angle_stochastic_gradient_descent(x1, x2, y):
    # batch = 15 #batch size
    alpha = 0.00001  # learning rate
    lamb = 0.09
    iter = 9999
    # initializing the learning rates to random values between 0 & 1
    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    # grabbing x0 as np.ones for future use
    x0 = np.ones(x1.shape[0])
    cost = [0 for i in range(iter)]
    w0_list = [1 for i in range(iter)]
    w1_list = [1 for i in range(iter)]
    w2_list = [1 for i in range(iter)]
    h = hypothesis(w0, w1, w2, x1, x2)
    for itr in range(0, iter):
        i = random.randint(0, y.shape[0] - 1)
        w0 = w0 - alpha * summa(h, y.shape[0], y, x0) - alpha * lamb * np.sign(w0) * 0.5
        w1 = w1 - alpha * summa(h, y.shape[0], y, x1) - alpha * lamb * np.sign(w1) * 0.5
        w2 = w2 - alpha * summa(h, y.shape[0], y, x2) - alpha * lamb * np.sign(w2) * 0.5
        w0_list[itr] = w0
        w1_list[itr] = w1
        w2_list[itr] = w2
        wsum = wabs(w0,w1,w2)
        h = hypothesis(w0, w1, w2, x1, x2)
        cost[itr] = (1 * 0.5 * mse(h, y, y.shape[0])/x1.shape[0] + lamb * wsum) * 0.5
    return w0, w1, w2, cost, w1_list, w2_list
