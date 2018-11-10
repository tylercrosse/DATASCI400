"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""
""" Replace an outlier."""
 
# Import the Numpy library
import numpy as np
# Create an array with the data
x = np.array([2, 1, 1, 99., 1, 5, 3, 1, 4, 3])
# => array([  2.,   1.,   1.,  99.,   1.,   5.,   3.,   1.,   4.,   3.])
#########################

# calculate the limits for values that are not outliers. 
LimitHi = np.mean(x) + 2*np.std(x)
# => 9.6906181234681821
LimitLo = np.mean(x) - 2*np.std(x)
# => -46.062035789317619

########################
# Create Flag for values outside of limits
FlagBad = (x < LimitLo) | (x > LimitHi)
# Uses bitwise or (|) https://wiki.python.org/moin/BitwiseOperators

# present the flag
FlagBad
########################

# Replace outlieres with mean of the whole array
# "Imputation" - n statistics, imputation is the process of replacing missing data with substituted values
x[FlagBad] = np.mean(x)

# See the values of x
x
#######################

# FlagGood is the complement of FlagBad
FlagGood = ~FlagBad

# Replace outliers with the mean of non-outliers
x[FlagBad] = np.mean(x[FlagGood])

# See the values of x
x
#######################
# Get the Sample data
x = np.array([2, 1, 1, 99., 1, 5, 3, 1, 4, 3])

# Replace outliers with the median of the whole array
x[FlagBad] = np.median(x)

# See the values of x
x
########################