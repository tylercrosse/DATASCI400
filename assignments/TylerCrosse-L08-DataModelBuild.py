""""
Lesson 08 Assignment
Tyler Crosse

Create a new Python script that includes the following.  Some of the items can be copied from your previous assignments:

1. Short narrative on the data preparation for your chosen data set from Milestone 2
2. Import statements for libraries and your data set
3. Show data preparation.  Normalize some numeric columns, one-hot encode some categorical columns with 3 or more categories, remove or replace missing values, remove or replace some outliers.
4. Specify an appropriate column as your expert label for a classification.  (include decision comments)
5. K-Means based on some of your columns, but excluding the expert label.  Add the cluster labels to your dataset.
6. Split the data set into training and testing sets (include decision comments)
7. Create a classification model for the expert label (include decision comments)
8. Write out to a csv a dataframe of predicted and actual values 
9. Determine accuracy, which is the number of correct predictions divided by the total number of predictions (include brief preliminary analysis commentary)
10. Comments explaining the code blocks. 
11. The grader must be able to execute your script on their computer using only the run file (F5) button in spyder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Read in the data from a freely available source on the internet
def _read_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    Adult = pd.read_csv(url, header=None)
    Adult.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
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
Consolidate education data
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
One-hot encode relationship data
"""
def _one_hot_relationship(Adult):
    Adult.loc[:, "married"] = Adult.loc[:, "relationship"].map({" Not-in-family":0, " Unmarried":0, " Own-child":0, " Other-relative":0, " Husband":1, " Wife":1})
    return Adult

"""
One-hot encode income data
"""
def _one_hot_income(Adult):
    Adult.loc[:, "over50K"] = np.where(Adult["income"] == " <=50K", 0, 1)
    return Adult

"""
K means, given two columns of data, normalizes the data, calculated K-means and plots it
"""
def _k_means(col_1, col_2, n_clusters):
    X = pd.DataFrame()
    X.loc[:, 0] = _z_normalize_values(col_1)
    X.loc[:, 1] = _z_normalize_values(col_2)
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X.loc[:, 0], X.loc[:, 1], c=y_kmeans)
    plt.show()

"""
Remove obsolete columns
"""
def _remove_obsolete(Adult, column_name):
    return Adult.drop(column_name, axis=1)

"""
Split the data
"""
def _split_data(data, y_data, test_size):
    return train_test_split(data, y_data, test_size=test_size)

"""
Predicts the values using a model and compares them against the target values
"""
def _test_accuracy(model, testing_features, testing_targets):
    predictions = model.predict(testing_features)
    return accuracy_score(testing_targets, predictions)

"""
Peform a Linear Regession
"""
def _linear_regression(training_features, training_targets):
    regr = LinearRegression()
    regr.fit(training_features, training_targets)
    return regr

"""
Decision Tree classifier
"""
def _decision_tree(training_features, training_targets):
    clf = DecisionTreeClassifier()
    clf.fit(training_features, training_targets)
    return clf

"""
Random forest classifier
"""
def _random_forest(training_features, training_targets):
    clf = RandomForestClassifier()
    clf.fit(training_features, training_targets)
    return clf

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
#    Adult["capital-gain"] = _handle_outliers(Adult["capital-gain"])
#    Adult["capital-loss"] = _handle_outliers(Adult["capital-loss"])
    # Adult = _impute_missing_numeric(Adult,  # There is no missing numeric data
    Adult["age"] = _z_normalize_values(Adult["age"])
    Adult = _consolidate_education_categories(Adult)
    Adult = _one_hot_education(Adult)
    Adult = _one_hot_relationship(Adult)
    Adult = _one_hot_income(Adult)
    Adult = _remove_obsolete(Adult, "education")
    Adult = _remove_obsolete(Adult, "income")
    _k_means(Adult["married"], Adult["over50K"], 2)
    Numeric = Adult.filter(["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "hs-no-grad", "hs-grad", "post-hs", "bach", "post-bach", "married", "over50K"], axis=1)
    X_train, X_test, y_train, y_test = _split_data(Numeric, Numeric["over50K"], 0.3)
    
    regr = _linear_regression(X_train, y_train)
#    dt_clf = _decision_tree(X_train, y_train)
#    rf_clf = _random_forest(X_train, y_train)
    print("Linear Regression Accuracy: %s%%" % (100*_test_accuracy(regr, X_test, y_test)))
#    print("Decision Tree Accuracy: %s%%" % (100*_test_accuracy(dt_clf, X_test, y_test)))
#    print("Random Forest Accuracy: %s%%" % (100*_test_accuracy(rf_clf, X_test, y_test)))
    Predicted = pd.DataFrame()
    Predicted.loc[:, "predicted"] = regr.predict(X_test)
    Predicted.loc[:, "Actual"] = y_test
    _write_to_file(Predicted, "TylerCrosse-L08-PredictedActual.csv")
