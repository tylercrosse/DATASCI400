"""
Lesson 07
Tyler Crosse

Create a text document listing possible questions your data set can answer. Include the following components in your write-up:

1. Description of your Milestone 3 dataset (May be the same as Milestone 2)
number of observations and attributes datatype, distribution, and a comment on each attribute
2. Source citation for your data set
3. Ask at least 1 yes-no or binary-choice question (Does it...? Is it...?)
4. Ask at least 1 non-binary question (What is...? How many...? When does...?)
5. Perform a K-Means with sklearn using some or all of your attributes.
6. Include at least one categorical column and one numeric attribute. 
7. Normalize the attributes prior to K-Means.
8. Add the cluster label to the dataset.
"""

import pandas as pd
import numpy as np

# Read in the data from UCI Machine Learning Dataset https://archive.ics.uci.edu/ml/datasets/Adult
def _read_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    Adult = pd.read_csv(url, header=None)
    Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    return Adult

"""
Discussion

The dataset used in this workplace scenario is the Adult dataset that was orginally extracted by Barry Becker from the 1994 Census database. It contains 14 attributes and 32,561 obesrvations. The attributes are as follows:
    Age, the age of the adult, is an integer with a mean 38.58, a standar deviatoin of 13.64, a min of 17 and a max of 90.
    Workclass, the working class of the adult, values consist of: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, and Never-worked. 
    Fnlwgt, the final weight as prepared as indendent controls by the Census Bureau, is an integer. 
    Education-num, is the number years of educaiton recieived, integer, with a mean of 10.08, a standard deviation of 2.57, a min of 1, and a max of 16.
    Education, the highest level of education recieved, the values consist of: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
    Marital-status, values consist of: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
    Occupation, the values consist of: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
    Relationship, the relation ship status, the values consist of: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
    Race, the values consist of: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
    Sex, the values consist of: Female, Male. 
    Capital-gain, is an interger representing the money earned from capital assets, with a mean of 1077.65, a standard deviation of 7385.29, a min of 0, a max of 99999, and the first 3 quartiles are all 0 .
    Capital-loss, is an interger representing the money lost from capital assets, with a mean of  87.30, a standard deviation of 402.96, a min of 0, max of 4356, and the first 3 quartiles are all 0.
    Hours-per-week, hours worked per week, integer, a mean of 40.44, a standard deviation of 12.35, a min of 1, and a max of 99.
    Native-country: the values consist of a list of countries.
    Income, whether or not the adult made above or below 50k.

There a number of questions that could be asked about the data set. Does gender effect earning over 50k. Is there a correlation between marital status and earning over 50k? What is the average hours per week worked by individuals earning over 50k vs those earning under 50k? How does occupation affect hours worked per week? Does race play an impact on the highest lelve of education or earning?

"""
