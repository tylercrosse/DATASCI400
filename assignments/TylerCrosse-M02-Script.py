#!/usr/bin/env python
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

# Read in the data from a freely available source on the internet
def _read_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    Adult = pd.read_csv(url, header=None)
    Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    return Adult


"""
Account for outlier values in numeric columns
"""
def _handle_outliers():
    print("Handle outliers")

"""
Replace missing numeric data
"""
def _impute_missing_numeric():
    print("Impute missing")

"""
Normalize numeric values
"""
def _normalize_values():
    print("normalize")

"""
Bin numeric variables
"""
def _bin():
    print("bin")

"""
Consolidate categorical data
"""
def _consolidate_categories():
    print("consolidate")

"""
One-hot encode categorical data
"""
def _one_hot():
    print("one hot")

"""
Remove obsolete columns
"""
def _remove_obsolete():
    print("remove obsolete")

"""
Write dataframe to file
"""
def _write_to_file(df, file_name):
    df.to_csv(file_name, sep=",", index=False)


"""
Main block
"""
if __name__ == "__main__":
    Adult = _read_data()
    print(Adult.head())

