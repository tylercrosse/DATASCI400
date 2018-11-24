#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 09:54:19 2018

@author: tcrosse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  
from copy import deepcopy

def FindAll(x, y): # find all elements in x that are equal to y
    N = len(x)
    z = np.zeros(N, dtype = bool)
    
    for i in range(N):
        if x[i] == y:
            z[i] = True
    
    ind = z * (np.array(range(N)) + np.ones(N, dtype = int))
    ind = ind[ind > 0]
    n = len(ind)    
    return ind - np.ones(n, dtype = int)

def kmeans(X, C, k, th):
    if k < 2:
        print('k needs to be at least 2!')
        return
    if (th <= 0.0) or (th >= 1.0):
        print('th values are beyond meaningful bounds')
        return    
    
    N, m = X.shape # dimensions of the dataset
    Y = np.zeros(N, dtype=int) # cluster labels
#    C = np.random.uniform(0, 1, [k,m]) # centroids
    d = th + 1.0
    dist_to_centroid = np.zeros(k) # centroid distances
    
    while d > th:
        C_ = deepcopy(C)
        
        for i in range(N): # assign cluster labels to all data points            
            for j in range(k): 
                dist_to_centroid[j] = np.sqrt(sum((X[i,] - C[j,])**2))                
            Y[i] = np.argmin(dist_to_centroid) # assign to most similar cluster            
            
        for j in range(k): # recalculate all the centroids
            ind = FindAll(Y, j) # indexes of data points in cluster j
            n = len(ind)            
            if n > 0: C[j] = sum(X[ind,]) / n
        
        d = np.mean(abs(C - C_)) # how much have the centroids shifted on average?
        
    return Y, C

# Q4
#X = np.array([[1,1], [1,2], [2,1], [2,2], [0,0], [0,1], [1,0], [1.5, 1.5], 
#        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5]])
    
# Q5
#C = np.array([[1,1], [2,2], [11,1], [1,11]])
#X = np.array([[1,1], [1,2], [2,1], [2,2], [10,1], [11,2], [12,1], [12,2], 
#        [1,11], [1,12], [2,11], [2,12]])
    
# Q6
C = np.array([[1,1], [2,2]])
X = np.array([[1,1], [1,2], [2,1], [2,2], [10,1], [11,2], [12,1], [12,2], [1,11], [1,12], [2,11], [2,12]])
th = 0.0001
k = 3

#plt.scatter(X[:, 0], X[:, 1])
##plt.show()
#
Y, C = kmeans(X, C, k, th)  
#
plt.scatter(X[:,0], X[:,1], c=Y, cmap='rainbow')
plt.show()

#plt.sc
#
#        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
#                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
#        plt.scatter(ClusterCentroids.loc[LabelNumber,0], ClusterCentroids.loc[LabelNumber,1], s=200, c="black", marker=marker)
#    plt.title(Title)
#    plt.show()



