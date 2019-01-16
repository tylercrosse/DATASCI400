#!/usr/bin/env python3
"""
Milestone 3 Assignment
Tyler Crosse


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import *

"""
Preparation of Data Set

Criteria: 1. Source citation for your data set 2. Data read from an easily and freely accessible source 3. Number of observations and attributes 4. Data types 5. Distribution of numerical variables 6. Distribution of categorical variables 7. A comment on each attribute 8. Removing cases with missing data 9. Removing outliers 10. Imputing missing values 11. Decoding 12. Consolidation 13. One-hot encoding 14. Normalization
"""
def _prepare_data():
    # 1. Source citation for your data set
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    print('\n[prepare 1.] Data Source:', url)
    
    # 2. Data read from an easily and freely accessible source
    Adult = pd.read_csv(url, header=None)
    Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    # 3. Number of observations and attributes (including index)
    print('\n[prepare 3.] {observations} observations and {attributes} attributes'.format(observations=len(Adult), attributes=len(Adult.columns)))
    
    # 4. Data types
    print('\n[prepare 4.] Data types:\n', Adult.dtypes)
    
    # 5. Distribution of numerical variables
    print('\n[prepare 5.] Distribution of numerical variables:\n', Adult.std())
    
    # 6. Distribution of categorical variables
    print('\n[prepare 6.] Distribution of categorical variables:')
    for i in Adult.columns:
        if type(Adult[i][1]) == str:
            Adult[i].value_counts().plot(kind='bar')
            plt.show()
            
    # 7. A comment on each attribute
    print("""\n[prepare 7.] A comment on each attribute:\n    Age, the age of the adult, is an integer with a mean 38.58, a standar deviatoin of 13.64, a min of 17 and a max of 90.
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
    Income, whether or not the adult made above or below 50k.""")
    
    # 8. Removing cases with missing data - workclass, occupation & native-country all had missing data. Removing the rows with missing occupation also removed the missing workclass values. 
    # 9. Removing outliers - there were no attributes where removing the outliers improved the data.
    # 10. Imputing missing values - the missing values for native-country were imputed as "outside-US".
    Adult = Adult[Adult["occupation"] != " ?"]
    Adult.loc[Adult.loc[:, "native-country"] != " United-States", "native-country"] = "outside-US"
    
    # 11. Decoding - all of the categorical data was already decoded.
    # 12. Consolidation
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
    
    # 13. One-hot encoding
    Adult.loc[:, "hs-no-grad"] = (Adult.loc[:, "education"] == " HS-no-grad").astype(int)
    Adult.loc[:, "hs-grad"] = (Adult.loc[:, "education"] == " HS-grad").astype(int)
    Adult.loc[:, "post-hs"] = (Adult.loc[:, "education"] == " Post-HS").astype(int)
    Adult.loc[:, "bach"] = (Adult.loc[:, "education"] == " Bachelors").astype(int)
    Adult.loc[:, "post-bach"] = (Adult.loc[:, "education"] == " Post-Bach").astype(int)
    
    Adult.loc[:, "married"] = Adult.loc[:, "relationship"].map({" Not-in-family":0, " Unmarried":0, " Own-child":0, " Other-relative":0, " Husband":1, " Wife":1})
    
    Adult.loc[:, "workclass-encode"] = Adult.loc[:, "workclass"].map({" Private": 0, " State-gov": 1, " Federal-gov": 2, " Self-emp-not-inc": 3, " Self-emp-inc": 4, " Local-gov": 5, " Without-pay": 6}).fillna(-1).astype(int)
    
    Adult.loc[:, "job-encode"] = Adult.loc[:, "occupation"].map({"Priv-house-serv": 0, " Protective-serv": 1, " Handlers-cleaners": 2, " Machine-op-inspct": 3, " Adm-clerical": 4, " Farming-fishing": 5, " Transport-moving": 6, " Craft-repair": 7, " Other-service": 8, " Tech-support": 9, " Sales": 10, " Exec-managerial": 11, " Prof-specialty": 12, " Armed-Forces": 13}).fillna(-1).astype(int)
    
    Adult.loc[:, "from-us"] = np.where(Adult.loc[:, "native-country"] == " United-States", 0, 1)

    Adult.loc[:, "over-50k"] = np.where(Adult.loc[:, "income"] == " <=50K", 0, 1)

    # 14. Normalization - done as preperation for K means
    
    return Adult

"""
Unsupervised Learning

Criteria: 1. Perform a K-Means with sklearn using some of your attributes. 2. Include at least one categorical column and one numeric attribute. Neither may be a proxy for the expert label in supervised learning. 3. Normalize the attributes prior to K-Means or justify why you didn't normalize. 4. Add the cluster label to the data set to be used in supervised learning
"""
def _unsupervised_learning(Adult):
    # 1. Perform a K-Means with sklearn using some of your attributes.
    # 2. Include at least one categorical column and one numeric attribute. Neither may be a proxy for the expert label in supervised learning.
    # 3. Normalize the attributes prior to K-Means or justify why you didn't normalize.
    
    def _z_normalize(vals):
        return (vals - np.mean(vals))/np.std(vals)
    
    def _k_means(df, col_1_name, col_2_name, n_clusters):
        X = pd.DataFrame()
        X.loc[:, 0] = _z_normalize(df[col_1_name])
        X.loc[:, 1] = _z_normalize(df[col_2_name])
        kmeans = KMeans(n_clusters)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        plt.scatter(X.loc[:, 0], X.loc[:, 1], c=y_kmeans)
        plt.xlabel(col_1_name)
        plt.ylabel(col_2_name)
        plt.show()
        return y_kmeans
     
    print("\n[unsupervised 1.-3.] Perform K-means with sklearn, including at least one categorical column and one numeric attribute, normalize the attributes prior to K-means")
    y_kmeans = _k_means(Adult, "hours-per-week", "job-encode", 2)

    # 4. Add the cluster label to the data set to be used in supervised learning
    Adult.loc[:, "cluster"] = y_kmeans
    
    return Adult

"""
Supervised Learning

Criteria: 1. Ask a binary-choice question that describes your classification. Write the question as a comment. 2. Split your data set into training and testing sets using the proper function in sklearn. 3. Use sklearn to train two classifiers on your training set, like logistic regression and random forest. 4. Apply your (trained) classifiers to the test set. 5. Create and present a confusion matrix for each classifier. Specify and justify your choice of probability threshold. 6. For each classifier, create and present 2 accuracy metrics based on the confusion matrix of the classifier. 7. For each classifier, calculate the ROC curve and it's AUC using sklearn. Present the ROC curve. Present the AUC in the ROC's plot.
"""
def _supervised_learning(Adult):
    # 1. Ask a binary-choice question that describes your classification. Write the question as a comment.
    print("\n[supervised 1.] Can marital status be used to predict whether or not an individual makes over 50k a year?")

    # 2. Split your data set into training and testing sets using the proper function in sklearn.
    X = Adult[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "hs-no-grad", "hs-grad", "post-hs", "bach", "post-bach", "married", "job-encode", "from-us", "cluster"]]
    y = Adult["over-50k"]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 3. Use sklearn to train two classifiers on your training set, like logistic regression and random forest.
    log_clf = LogisticRegression()
    log_clf.fit(x_train, y_train)
    
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(x_train, y_train)
    
    estimators = 10 # number of trees parameter
    mss = 2 # mininum samples split parameter
    rf_clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss)
    rf_clf.fit(x_train, y_train)

    # 4. Apply your (trained) classifiers to the test set.
    log_pred = log_clf.predict(x_test)
    dt_pred = dt_clf.predict(x_test)
    rf_pred = rf_clf.predict(x_test)

    # 5. Create and present a confusion matrix for each classifier. Specify and justify your choice of probability threshold.    
    print("\n[supervised 5.] create and present a confusion matrix for each classifier")
    baseline = pd.DataFrame(y_test)
    baseline["over-50k"] = 0
    tn, fp, fn, tp = confusion_matrix(y_test, baseline).ravel()
    threshold = (tp + tn)/(tp + tn + fp + fn)
    print("\n The threshold is %s because that was determined to be the baseline probability" % threshold)
    
    log_k = pd.DataFrame(confusion_matrix(y_test, log_pred))
    print("\nLogistic Regression confusion matrix:\n", log_k)
    dt_k = pd.DataFrame(confusion_matrix(y_test, dt_pred))
    print("\nDecision Tree confusion matrix:\n", dt_k)
    rf_k = pd.DataFrame(confusion_matrix(y_test, rf_pred))
    print("\nRandom Forest confusion matrix:\n", rf_k)

    # 6. For each classifier, create and present 2 accuracy metrics based on the confusion matrix of the classifier.
    print("\n[supervised 6.] present accuracy metrics\n")
    print("Logistic Regression:\n", classification_report(log_pred, y_test, target_names=["<=50k", ">50k"]))
    print("Decision Tree:\n", classification_report(log_pred, y_test, target_names=["<=50k", ">50k"]))
    print("Random Forest:\n", classification_report(rf_pred, y_test, target_names=["<=50k", ">50k"]))

    # 7. For each classifier, calculate the ROC curve and it's AUC using sklearn. Present the ROC curve. Present the AUC in the ROC's plot.
    
    # Adapted from example on https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    def _plot_roc(y_test, y_score, classifier_name):
        fpr, tpr, th = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange",
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for %s' % classifier_name)
        plt.legend(loc="lower right")
        plt.show()
        
    print("\n[supervised 7.] ROC curve and AUC")
    _plot_roc(y_test, log_clf.fit(x_train, y_train).decision_function(x_test), "Logistic Regression")
    _plot_roc(y_test, dt_clf.fit(x_train, y_train).predict_proba(x_test)[:,1], "Decision Tree")
    _plot_roc(y_test, rf_clf.fit(x_train, y_train).predict_proba(x_test)[:,1], "Random Forest")
    
    return ''


if __name__ == "__main__":
    Adult = _prepare_data()
    Adult = _unsupervised_learning(Adult)
    _supervised_learning(Adult)
