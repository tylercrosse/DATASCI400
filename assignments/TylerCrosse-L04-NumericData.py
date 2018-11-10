#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson 04 Assignment

Create a new Python script that includes the following for your Milestone 2 data set:
- Import statements
- Load your dataset
- Assign reasonable column names, the data set description
- Median imputation of the missing numeric values
- Outlier replacement if applicable
- Histogram of a numeric variable. Use plt.show() after each histogram
- Create a scatterplot. Use plt.show() after the scatterplot
- Determine the standard deviation of all numeric variables. Use print() for each 
  standard deviation
- Comments explaining the code blocks
- Summary comment block on how the numeric variables have been treated: which 
  ones had outliers, required imputation, distribution, removal of rows/columns.

"""

import pandas as pd
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
Adult = pd.read_csv(url, header=None)
Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Histogram of a numeric variable. Use plt.show() after each histogram
age = Adult.age
age.hist()
plt.show()

# Print the standard deviation for each numeric variable
print(Adult.std())

"""
There didn't appear to be any data that was missing or an outlier as a result of 
the collection methods. Both 'capital-gain' and 'capital-loss' had a lot of rows 
of zeros and a fairly large range of values but that seems to be intentional and 
accurate. 
"""
