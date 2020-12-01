import numpy as np
from functions_v import *
from functions import *
def q5(x1, x2, y, X1, X2, Y):
    x0 = np.ones(x1.shape)
    X_train = np.column_stack((x0, x1, x2))
    # X_train = np.concatenate((np.ones(shape = (X_train.shape[0],1)),X_train) ,axis = 1)
    w = np.random.randn(1, 3)
    print("BATCH GRADIENT DESCENT")
    print(" ")
    w, cost, w_list = batch_gradient_descent_v(X_train, y, w)
    print("W0 :", w[0][0])
    print("W1 :", w[0][1])
    print("W2 :", w[0][2])
    print(" ")
    #print(cost)
    plot2d("cost", "iterations", cost, "Batch-Gradient-Descent")
    #print(" ")
    #plot3d(cost, w1_list, w2_list, "cost", "w1", "w2", "Batch-Gradient-Descent")
    #print(" ")
    #pred(w0, w1, w2, X1, X2, Y)