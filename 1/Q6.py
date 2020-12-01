import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time


# Method to find the euclidean distance between centroid and points
def distance(data, centroid):
    return np.sqrt(np.sum((data - centroid) ** 2))


def cluster(item, centroids):
    a = distance(item, centroids[0])
    b = distance(item, centroids[1])
    c = distance(item, centroids[2])
    if min(a, b, c) == a:
        return 0
    elif min(a, b, c) == b:
        return 1
    else:
        return 2


def KMeansCluster(X, k):
    #    Taking 2 random points as centriod as the number of clusters is 2
    random_indices = np.random.choice(len(X), 3)
    #    Intitalising the centroids
    centroids = X[random_indices]
    #    Initialising the number of iterations
    iters = 1000
    clusters = np.zeros(len(X))

    #    Clustering the points into cluster 1 if the distance between
    #    the point and centroid 1  is less than distance between
    #    the point and centroid 2

    for it in range(iters):
        for i in range(len(X)):
            clusters[i] = cluster(X[i], centroids)
            # if distance(X[i], centroids[0]) < distance(X[i], centroids[1]) :
            #	clusters[i] = 0
            # else:
            #	clusters[i] = 1

    for i in range(k):
        pts = [X[j] for j in range(len(X)) if clusters[j] == i]
        centroids[i] = np.mean(pts, axis=0)

    return centroids, clusters


# Reading the data from the file
dataframe = pd.read_csv('data2.csv', header=None)

# Dividing into feature vectors
dataframe.columns = ['x1', 'x2', 'x3', 'x4']
feature1 = dataframe['x1'].values
feature2 = dataframe['x2'].values
feature3 = dataframe['x3'].values
feature4 = dataframe['x4'].values

# Naming the columns of the feature matrix
X = np.array(list(zip(feature1, feature2, feature3, feature4)))
# initialising Number of Clusters
k = 3

centroids, clusters = KMeansCluster(X, k)
print(centroids)

# Plotting the graphs seaprately for the respective features
plt.figure()
plt.scatter(np.arange(len(X)), X[:, 0], c=clusters.flatten())
plt.title('FEATURE 1')
plt.show()
plt.figure()
plt.scatter(np.arange(len(X)), X[:, 1], c=clusters.flatten())
plt.title('FEATURE 2')
plt.show()
plt.figure()
plt.scatter(np.arange(len(X)), X[:, 2], c=clusters.flatten())
plt.title('FEATURE 3')
plt.show()
plt.figure()
plt.scatter(np.arange(len(X)), X[:, 3], c=clusters.flatten())
plt.title('FEATURE 4')
plt.show()