def q4(x1,x2,y,X1,X2,Y):
    print("BATCH GRADIENT DESCENT (Least Angle Regression)")
    print(" ")
    w0, w1, w2, cost, w1_list, w2_list = least_angle_regression_batch(x1, x2, y)
    print("W0 :", w0)
    print("W1 :", w1)
    print("W2 :", w2)
    print(" ")
    plot2d("cost", "iterations", cost, "Batch-Gradient-Descent (Least Angle Regression)")
    print(" ")
    plot3d(cost, w1_list, w2_list, "cost", "w1", "w2", "Batch-Gradient-Descent (Least Angle Regression)")
    print(" ")
    pred(w0, w1, w2, X1, X2, Y)
    time.sleep(5)
    print("MINI BATCH GRADIENT DESCENT (Least Angle Regression)")
    print(" ")
    w0, w1, w2, cost, w1_list, w2_list = least_angle_mini_batch(x1, x2, y)
    print("W0 :", w0)
    print("W1 :", w1)
    print("W2 :", w2)
    print(" ")
    plot2d("cost", "iterations", cost, "Mini-Batch-Gradient-Descent (Least Angle Regression)")
    print(" ")
    plot3d(cost, w1_list, w2_list, "cost", "w1", "w2", "Mini-Batch-Gradient-Descent (Least Angle Regression)")
    print(" ")
    pred(w0, w1, w2, X1, X2, Y)
    print(" ")
    print(" ")
    print(" ")
    time.sleep(5)
    print("STOCHASTIC GRADIENT DESCENT (Least Angle Regression)")
    w0, w1, w2, cost, w1_list, w2_list = least_angle_stochastic_gradient_descent(x1, x2, y)
    print("W0 :", w0)
    print("W1 :", w1)
    print("W2 :", w2)
    print(" ")
    plot2d("cost", "iterations", cost, "Stochastic-Gradient-Descent (Least Angle Regression)")
    print(" ")
    plot3d(cost, w1_list, w2_list, "cost", "w1", "w2", "Stochastic-Gradient-Descent (Least Angle Regression)")
    print(" ")
    pred(w0, w1, w2, X1, X2, Y)
    time.sleep(5)