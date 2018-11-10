"""
Lesson 05 Assignment
Tyler Crosse

Create a new Python script that includes the following:

1. Import statements
2. Loading your dataset
3. Normalize one variable
4. Bin one variable
5. Decoding categorical data
6. Imputing missing values
7. Consolidating categories if applicable
8. One-hot encoding (dummy variables) for a categorical column with more than 2 categories
9. New columns created, obsolete deleted if applicable
10. Plots for 1 or more categories
11. Comments explaining the code blocks
12. Summary comment block on how the categorical data has been treated: decoded, imputed, consolidated, dummy variables created.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
Adult = pd.read_csv(url, header=None)
Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# 3. Normalize one variable
d = Adult.loc[:, 'age']
dz = (d - np.mean(d))/np.std(d)
dmm = (d - np.min(d))/(np.max(d) - np.min(d))

plt.hist(d, bins = 20)
plt.title('Original Distribution of age')
plt.ylabel('number of adults')
plt.xlabel('age')
plt.show()

plt.hist(dz, bins = 20)
plt.title('Z-normalization of age')
plt.ylabel('number of adults')
plt.xlabel('Z-normalized age')
plt.show()

plt.hist(dmm, bins = 20)
plt.title('min-max-normalization of age')
plt.ylabel('number of adults')
plt.xlabel('min-max normalized age')
plt.show()

# 4. Bin one variable
# exploring different bin sizes
plt.hist(Adult['hours-per-week'], bins = 3)
plt.title('3 equal-width bins')
plt.ylabel('number of adults')
plt.xlabel('hours per week')
plt.show()

plt.hist(Adult['hours-per-week'], bins = 6)
plt.title('6 equal-width bin')
plt.ylabel('number of adults')
plt.xlabel('hours per week')
plt.show()

# bin
x = Adult.loc[:, 'hours-per-week']
NumberOfBins = 3
BinWidth = (max(x) - min(x))/NumberOfBins
MinBin1 = 0
MaxBin2 = min(x) + 2 * BinWidth
MaxBin3 = min(x) + 3 * BinWidth
MaxBin4 = min(x) + 4 * BinWidth

Binned_EqF = np.array([' ']*len(x)) # Empty starting point for equal-frequency-binned array
Binned_EqF[(MinBin1 < x) & (x <= BinWidth)] = 'L' # Low
Binned_EqF[(BinWidth < x) & (x <= MaxBin2)] = 'M' # Med
Binned_EqF[(MaxBin2 < x) & (x  < MaxBin3)] = 'H' # High

# or you can use the pandas cut method https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html
Binned_EqF_cut = pd.cut(Adult['hours-per-week'], 3, labels=['L', 'M', 'H'])

# 5. Decoding categorical data => N/A
# 6. Imputing missing values
# 7. Consolidating categories if applicable

# consolidate and impute native countries
Adult.loc[Adult.loc[:, 'native-country'] != ' United-States', 'native-country'] = ' Outside-US'

# consolidate education categories
Adult.loc[Adult.loc[:, 'education'] == ' 12th', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' 11th', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' 10th', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' 9th', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' 7th-8th', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' 5th-6th', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' 1st-4th', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' Preschool', 'education'] = ' HS-no-grad'
Adult.loc[Adult.loc[:, 'education'] == ' Some-college', 'education'] = ' Post-HS'
Adult.loc[Adult.loc[:, 'education'] == ' Assoc-voc', 'education'] = ' Post-HS'
Adult.loc[Adult.loc[:, 'education'] == ' Assoc-acdm', 'education'] = ' Post-HS'
Adult.loc[Adult.loc[:, 'education'] == ' Prof-school', 'education'] = ' Post-HS'
Adult.loc[Adult.loc[:, 'education'] == ' Masters', 'education'] = ' Post-Bach'
Adult.loc[Adult.loc[:, 'education'] == ' Doctorate', 'education'] = ' Post-Bach'

# One-hot encoding for category with 2 categories
Adult['relationship'] = Adult['relationship'].map({' Not-in-family':0, ' Unmarried':0, ' Own-child':0, ' Other-relative':0, ' Husband':1, ' Wife':1})

# 8. One-hot encoding (dummy variables) for a categorical column with more than 2 categories
Adult.loc[:, 'hs-no-grad'] = (Adult.loc[:, 'education'] == ' HS-no-grad').astype(int)
Adult.loc[:, 'hs-grad'] = (Adult.loc[:, 'education'] == ' HS-grad').astype(int)
Adult.loc[:, 'post-hs'] = (Adult.loc[:, 'education'] == ' Post-HS').astype(int)
Adult.loc[:, 'bach'] = (Adult.loc[:, 'education'] == ' Bachelors').astype(int)
Adult.loc[:, 'post-bach'] = (Adult.loc[:, 'education'] == ' Post-Bach').astype(int)

# 9. New columns created, obsolete deleted if applicable
Adult = Adult.drop('education', axis=1)

# 10. Plots for 1 or more categories
Adult['native-country'].value_counts().plot(kind='bar')
plt.title('native country categories')
plt.ylabel('number of adults')
plt.show()

"""
Good candidates for normailzation were any of the continuous variables
[age, education-num, capital-gain, capital-loss, hours-per-week]
Age was choosen because it had a fairly normal distribution.
Hours per week were binned with equal size bins.
None of the categorical data started out encoded so this was n/a
The categories for native countries, education, and relationship were consolidated.
This also involved imputing values for native country.
Relationship and education were then one-hot encoded.
The categories for native-country were then plotted. 
"""