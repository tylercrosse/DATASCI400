"""
Milestone 2 Assignment
Tyler Crosse

Write a script to prepare a data set with known missing values.  The data could be the same data that you will use for a predictive model in milestone 3.  The script must run, without alterations, in the grader's environment.

1. Read in the data from a freely available source on the internet.  
2. Account for outlier values in numeric columns (at least 1 column).
3. Replace missing numeric data (at least 1 column).
4. Normalize numeric values (at least 1 column, but be consistent with numeric data).
5. Bin numeric variables (at least 1 column).
6. Consolidate categorical data (at least 1 column).
7. One-hot encode categorical data with at least 3 categories (at least 1 column).
8. Remove obsolete columns.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
Adult = pd.read_csv(url, header=None)
Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# 2. Account for outlier values in numeric columns (at least 1 column).


# 3. Replace missing numeric data (at least 1 column).


# 4. Normalize numeric values (at least 1 column, but be consistent with numeric data).


# 5. Bin numeric variables (at least 1 column).


# 6. Consolidate categorical data (at least 1 column).


# 7. One-hot encode categorical data with at least 3 categories (at least 1 column).


# 8. Remove obsolete columns.


# Write to file TylerCrosse-M02-Dataset.csv

