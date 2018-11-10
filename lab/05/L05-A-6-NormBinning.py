"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

from sklearn.preprocessing import *
import pandas as pd
import numpy as np

NB = 3 # number of bins
x = np.array([1,11,5,3,15,3,5,9,7,9,3,5,7,3,5,21])
X = pd.DataFrame(x)
# freq, bounds = np.histogram(x, NB) # one way of obtaining the boundaries of the bins
bounds = np.linspace(np.min(x), np.max(x), NB + 1) # more straight-forward way for obtaining the boundaries of the bins

# bin assigns a bin label to each value of array x
# bin cannot handle values of x that are less than the first bound or greater
# than the last bound
def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, object) # empty string array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= bounds[i-1])&(x < bounds[i])] = str(i)
    
    y[x == bounds[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y

# NORMALIZING
minmax_scale = MinMaxScaler().fit(X)
standardization_scale = StandardScaler().fit(X)
y = minmax_scale.transform(X)
z = standardization_scale.transform(X)
print ("\nScaled variable x using MinMax and Standardized scaling\n")
print (np.hstack((np.reshape(x, (16,1)), y, z)))

# BINNING
bx = bin(x, bounds)
print ("\n\nBinned variable x, for ", NB, "bins\n")
print ("Bin boundaries: ", bounds)
print ("Binned variable: ", bx)
