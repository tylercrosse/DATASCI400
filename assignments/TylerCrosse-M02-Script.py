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

# Read in the data from a freely available source on the internet
def _read_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    Adult = pd.read_csv(url, header=None)
    Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    return Adult


"""
Account for outlier values in numeric column
Removes outliers using boolean mask with limits of 2 std deviantions from mean

"""
def _handle_outliers(vals):
    limit_hi = np.mean(vals) + 2*np.std(vals)
    limit_lo = np.mean(vals) - 2*np.std(vals)
    flag_good = (vals >= limit_lo) & (vals <= limit_hi)
    return vals[flag_good]

"""
Replace missing numeric data
"""
def _impute_missing_numeric(df, column_name):
    return df[df[column_name] != "?"]


"""
Z Normalize numeric values
"""
def _z_normalize_values(vals):
    return (vals - np.mean(vals))/np.std(vals)


"""
Bin numeric variables

https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html
"""
def _bin(vals, bins, labels):
    return pd.cut(Adult["hours-per-week"], bins, labels=labels)

"""
Consolidate categorical data
"""
def _consolidate_education_categories(Adult):
    Adult.loc[Adult.loc[:, "education"] == " 12th", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " 11th", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " 10th", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " 9th", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " 7th-8th", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " 5th-6th", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " 1st-4th", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " Preschool", "education"] = " HS-no-grad"
    Adult.loc[Adult.loc[:, "education"] == " Some-college", "education"] = " Post-HS"
    Adult.loc[Adult.loc[:, "education"] == " Assoc-voc", "education"] = " Post-HS"
    Adult.loc[Adult.loc[:, "education"] == " Assoc-acdm", "education"] = " Post-HS"
    Adult.loc[Adult.loc[:, "education"] == " Prof-school", "education"] = " Post-HS"
    Adult.loc[Adult.loc[:, "education"] == " Masters", "education"] = " Post-Bach"
    Adult.loc[Adult.loc[:, "education"] == " Doctorate", "education"] = " Post-Bach"
    return Adult

"""
One-hot encode categorical data
"""
def _one_hot_education(Adult):
    Adult.loc[:, "hs-no-grad"] = (Adult.loc[:, "education"] == " HS-no-grad").astype(int)
    Adult.loc[:, "hs-grad"] = (Adult.loc[:, "education"] == " HS-grad").astype(int)
    Adult.loc[:, "post-hs"] = (Adult.loc[:, "education"] == " Post-HS").astype(int)
    Adult.loc[:, "bach"] = (Adult.loc[:, "education"] == " Bachelors").astype(int)
    Adult.loc[:, "post-bach"] = (Adult.loc[:, "education"] == " Post-Bach").astype(int)
    return Adult

"""
Remove obsolete columns
"""
def _remove_obsolete_education(Adult):
    return Adult.drop("education", axis=1)

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
    Adult["capital-gain"] = _handle_outliers(Adult["capital-gain"])
    Adult["capital-loss"] = _handle_outliers(Adult["capital-loss"])
    # Adult = _impute_missing_numeric(Adult,  # There is no missing numeric data
    Adult["age"] = _z_normalize_values(Adult["age"])
    Adult["hours-per-week"] = _bin(Adult["hours-per-week"], 3, ["part-time", "full-time", "over-time"])
    Adult = _consolidate_education_categories(Adult)
    Adult = _one_hot_education(Adult)
    Adult = _remove_obsolete_education(Adult)
    _write_to_file(Adult, "TylerCrosse-M02-Dataset.csv")
