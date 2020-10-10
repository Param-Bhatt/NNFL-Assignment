import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time


def hypothesis_v(theta, X, Y):
  h = (np.matmul(X, np.transpose(theta)))
  for i in range(0, h.shape[0]):
      h[i] = h[i]-Y[i]
  return h

def batch_gradient_descent_v(x, y, w):
    iters = 300
    alpha = 0.0001
    h = hypothesis_v(w, x, y)
    cost = [0 for i in range(iters)]
    w_list = np.ones(shape = (x.shape[0],w.shape[1]))
    for i in range(0, iters):
        for j in range(0, w.shape[1]):
          w[0][j] = w[0][j] - (alpha *  np.sum(np.matmul(np.transpose(h),x[:,j])))
        for j in range(0,w.shape[1]):
          w_list[j][0] = w[0][j]
        h = hypothesis_v(w, x, y)
        cost[i] = (np.sum(np.square(h-y))*0.5)
        #cost[i] = 1 * `0.5 * mse(h, y, y.shape[0])
    return w, cost, w_list