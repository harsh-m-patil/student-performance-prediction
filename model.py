# -*- coding: utf-8 -*-
"""Student Performance Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ojbRX9dwOh0MfRi-BQ-HzgUHGNpXyZFw

#### Import Files
"""

# Commented out IPython magic to ensure Python compatibility.
# Import Files
import pandas as pd
import numpy as np
import os
import sys
import random
import time
import matplotlib
import matplotlib.pyplot as plt

from sklearn.svm import SVR
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from math import sqrt

# %matplotlib inline

# Importing the datasets

# Mathematics Data
student_mat = pd.read_csv('./data/student-mat.csv', sep=';')

# Portuguese Data
student_por = pd.read_csv("./data/student-por.csv", sep=';')

# Checking out few rows of Mathematics
student_mat.head()

# Checking out few rows of Portuguese
student_por.head()

# Number of records
print("Number of Mathematics Student Data:")
len(student_mat)

# Number of records
print("Number of Portuguese Student Data:")
len(student_por)

# Number of Attributes, should be 33 as specified at the data website
print("Attribute List:")
print(list(student_mat))
print()
print("Number of attributes: ", end="")
print(len(list(student_mat)))

# Number of Attributes, should be 33 as specified at the data website
print("Attribute List:")
print(list(student_por))
print()
print("Number of attributes: ", end="")
print(len(list(student_por)))

"""## Preprocessing the data to be made ready for analysis"""

# Copying the data, so orignal remains intact
student_mat_clean = student_mat.copy()
student_por_clean = student_por.copy()

"""### For Mathematics students"""

# School
# Unique Schools
print(student_mat.school.unique())
student_mat_clean.loc[student_mat_clean['school']=='GP', 'school'] = 1
student_mat_clean.loc[student_mat_clean['school']=='MS', 'school'] = 2
print(student_mat_clean.school.unique())
print()


# Gender
# Unique Genders
print(student_mat.sex.unique())
student_mat_clean.loc[student_mat_clean['sex']=='M', 'sex'] = 1
student_mat_clean.loc[student_mat_clean['sex']=='F', 'sex'] = 2
print(student_mat_clean.sex.unique())
print()



# Address
# Unique Address
print(student_mat.address.unique())
student_mat_clean.loc[student_mat_clean['address']=='R', 'address'] = 1
student_mat_clean.loc[student_mat_clean['address']=='U', 'address'] = 2
print(student_mat_clean.address.unique())
print()

# Family Size
# Unique Family Sizes
print(student_mat.famsize.unique())
student_mat_clean.loc[student_mat_clean['famsize']=='LE3', 'famsize'] = 1
student_mat_clean.loc[student_mat_clean['famsize']=='GT3', 'famsize'] = 2
print(student_mat_clean.famsize.unique())
print()



# Parent Status
# Unique Parent Statuses
print(student_mat.Pstatus.unique())
student_mat_clean.loc[student_mat_clean['Pstatus']=='A', 'Pstatus'] = 1
student_mat_clean.loc[student_mat_clean['Pstatus']=='T', 'Pstatus'] = 2
print(student_mat_clean.Pstatus.unique())
print()


# Mother's Job Status
# Unique Mothers' jobs
print(student_mat.Mjob.unique())
student_mat_clean.loc[student_mat_clean['Mjob']=='teacher', 'Mjob'] = 5
student_mat_clean.loc[student_mat_clean['Mjob']=='services', 'Mjob'] = 4
student_mat_clean.loc[student_mat_clean['Mjob']=='health', 'Mjob'] = 3
student_mat_clean.loc[student_mat_clean['Mjob']=='other', 'Mjob'] = 2
student_mat_clean.loc[student_mat_clean['Mjob']=='at_home', 'Mjob'] = 1
print(student_mat_clean.Mjob.unique())
print()


# Father's Job Status
# Unique Fathers' jobs
print(student_mat.Fjob.unique())
student_mat_clean.loc[student_mat_clean['Fjob']=='teacher', 'Fjob'] = 5
student_mat_clean.loc[student_mat_clean['Fjob']=='services', 'Fjob'] = 4
student_mat_clean.loc[student_mat_clean['Fjob']=='health', 'Fjob'] = 3
student_mat_clean.loc[student_mat_clean['Fjob']=='other', 'Fjob'] = 2
student_mat_clean.loc[student_mat_clean['Fjob']=='at_home', 'Fjob'] = 1
print(student_mat_clean.Fjob.unique())
print()


# Reasons
# Unique Reasons
print(student_mat.reason.unique())
student_mat_clean.loc[student_mat_clean['reason']=='reputation', 'reason'] = 4
student_mat_clean.loc[student_mat_clean['reason']=='course', 'reason'] = 3
student_mat_clean.loc[student_mat_clean['reason']=='home', 'reason'] = 2
student_mat_clean.loc[student_mat_clean['reason']=='other', 'reason'] = 1
print(student_mat_clean.reason.unique())
print()




# Guardians
# Unique Guardians
print(student_mat.guardian.unique())
student_mat_clean.loc[student_mat_clean['guardian']=='father', 'guardian'] = 3
student_mat_clean.loc[student_mat_clean['guardian']=='mother', 'guardian'] = 2
student_mat_clean.loc[student_mat_clean['guardian']=='other', 'guardian'] = 1
print(student_mat_clean.guardian.unique())
print()


# School Support
# Unique School Supports
print(student_mat.schoolsup.unique())
student_mat_clean.loc[student_mat_clean['schoolsup']=='yes', 'schoolsup'] = 2
student_mat_clean.loc[student_mat_clean['schoolsup']=='no', 'schoolsup'] = 1
print(student_mat_clean.schoolsup.unique())
print()

# Family Support
# Unique Family Supports
print(student_mat.famsup.unique())
student_mat_clean.loc[student_mat_clean['famsup']=='yes', 'famsup'] = 2
student_mat_clean.loc[student_mat_clean['famsup']=='no', 'famsup'] = 1
print(student_mat_clean.famsup.unique())
print()


# Paid
# Unique Pays
print(student_mat.paid.unique())
student_mat_clean.loc[student_mat_clean['paid']=='yes', 'paid'] = 2
student_mat_clean.loc[student_mat_clean['paid']=='no', 'paid'] = 1
print(student_mat_clean.paid.unique())
print()


# Activities
# Unique Activities
print(student_mat.activities.unique())
student_mat_clean.loc[student_mat_clean['activities']=='yes', 'activities'] = 2
student_mat_clean.loc[student_mat_clean['activities']=='no', 'activities'] = 1
print(student_mat_clean.activities.unique())
print()


# Nursery
# Unique Nursery status
print(student_mat.nursery.unique())
student_mat_clean.loc[student_mat_clean['nursery']=='yes', 'nursery'] = 2
student_mat_clean.loc[student_mat_clean['nursery']=='no', 'nursery'] = 1
print(student_mat_clean.nursery.unique())
print()


# Higher Education
# Unique Higher Education status
print(student_mat.higher.unique())
student_mat_clean.loc[student_mat_clean['higher']=='yes', 'higher'] = 2
student_mat_clean.loc[student_mat_clean['higher']=='no', 'higher'] = 1
print(student_mat_clean.higher.unique())
print()

# Internet
# Unique Internet Status
print(student_mat.internet.unique())
student_mat_clean.loc[student_mat_clean['internet']=='yes', 'internet'] = 2
student_mat_clean.loc[student_mat_clean['internet']=='no', 'internet'] = 1
print(student_mat_clean.internet.unique())
print()


# Relationship
# Unique Relationship Status
print(student_mat.romantic.unique())
student_mat_clean.loc[student_mat_clean['romantic']=='yes', 'romantic'] = 2
student_mat_clean.loc[student_mat_clean['romantic']=='no', 'romantic'] = 1
print(student_mat_clean.romantic.unique())
print()

"""### For Portuguese students"""

# School
# Unique Schools
print(student_por.school.unique())
student_por_clean.loc[student_por_clean['school']=='GP', 'school'] = 1
student_por_clean.loc[student_por_clean['school']=='MS', 'school'] = 2
print(student_por_clean.school.unique())
print()


# Gender
# Unique Genders
print(student_por.sex.unique())
student_por_clean.loc[student_por_clean['sex']=='M', 'sex'] = 1
student_por_clean.loc[student_por_clean['sex']=='F', 'sex'] = 2
print(student_por_clean.sex.unique())
print()


# Address
# Unique Address
print(student_por.address.unique())
student_por_clean.loc[student_por_clean['address']=='R', 'address'] = 1
student_por_clean.loc[student_por_clean['address']=='U', 'address'] = 2
print(student_por_clean.address.unique())
print()


# Family Size
# Unique Family Sizes
print(student_por.famsize.unique())
student_por_clean.loc[student_por_clean['famsize']=='LE3', 'famsize'] = 1
student_por_clean.loc[student_por_clean['famsize']=='GT3', 'famsize'] = 2
print(student_por_clean.famsize.unique())
print()


# Parent Status
# Unique Parent Statuses
print(student_por.Pstatus.unique())
student_por_clean.loc[student_por_clean['Pstatus']=='A', 'Pstatus'] = 1
student_por_clean.loc[student_por_clean['Pstatus']=='T', 'Pstatus'] = 2
print(student_por_clean.Pstatus.unique())
print()


# Mother's Job Status
# Unique Mothers' jobs
print(student_por.Mjob.unique())
student_por_clean.loc[student_por_clean['Mjob']=='teacher', 'Mjob'] = 5
student_por_clean.loc[student_por_clean['Mjob']=='services', 'Mjob'] = 4
student_por_clean.loc[student_por_clean['Mjob']=='health', 'Mjob'] = 3
student_por_clean.loc[student_por_clean['Mjob']=='other', 'Mjob'] = 2
student_por_clean.loc[student_por_clean['Mjob']=='at_home', 'Mjob'] = 1
print(student_por_clean.Mjob.unique())
print()


# Father's Job Status
# Unique Fathers' jobs
print(student_por.Fjob.unique())
student_por_clean.loc[student_por_clean['Fjob']=='teacher', 'Fjob'] = 5
student_por_clean.loc[student_por_clean['Fjob']=='services', 'Fjob'] = 4
student_por_clean.loc[student_por_clean['Fjob']=='health', 'Fjob'] = 3
student_por_clean.loc[student_por_clean['Fjob']=='other', 'Fjob'] = 2
student_por_clean.loc[student_por_clean['Fjob']=='at_home', 'Fjob'] = 1
print(student_por_clean.Fjob.unique())
print()


# Reasons
# Unique Reasons
print(student_por.reason.unique())
student_por_clean.loc[student_por_clean['reason']=='reputation', 'reason'] = 4
student_por_clean.loc[student_por_clean['reason']=='course', 'reason'] = 3
student_por_clean.loc[student_por_clean['reason']=='home', 'reason'] = 2
student_por_clean.loc[student_por_clean['reason']=='other', 'reason'] = 1
print(student_por_clean.reason.unique())
print()




# Guardians
# Unique Guardians
print(student_por.guardian.unique())
student_por_clean.loc[student_por_clean['guardian']=='father', 'guardian'] = 3
student_por_clean.loc[student_por_clean['guardian']=='mother', 'guardian'] = 2
student_por_clean.loc[student_por_clean['guardian']=='other', 'guardian'] = 1
print(student_por_clean.guardian.unique())
print()


# School Support
# Unique School Supports
print(student_por.schoolsup.unique())
student_por_clean.loc[student_por_clean['schoolsup']=='yes', 'schoolsup'] = 2
student_por_clean.loc[student_por_clean['schoolsup']=='no', 'schoolsup'] = 1
print(student_por_clean.schoolsup.unique())
print()

# Family Support
# Unique Family Supports
print(student_por.famsup.unique())
student_por_clean.loc[student_por_clean['famsup']=='yes', 'famsup'] = 2
student_por_clean.loc[student_por_clean['famsup']=='no', 'famsup'] = 1
print(student_por_clean.famsup.unique())
print()


# Paid
# Unique Pays
print(student_por.paid.unique())
student_por_clean.loc[student_por_clean['paid']=='yes', 'paid'] = 2
student_por_clean.loc[student_por_clean['paid']=='no', 'paid'] = 1
print(student_por_clean.paid.unique())
print()


# Activities
# Unique Activities
print(student_por.activities.unique())
student_por_clean.loc[student_por_clean['activities']=='yes', 'activities'] = 2
student_por_clean.loc[student_por_clean['activities']=='no', 'activities'] = 1
print(student_por_clean.activities.unique())
print()


# Nursery
# Unique Nursery status
print(student_por.nursery.unique())
student_por_clean.loc[student_por_clean['nursery']=='yes', 'nursery'] = 2
student_por_clean.loc[student_por_clean['nursery']=='no', 'nursery'] = 1
print(student_por_clean.nursery.unique())
print()


# Higher Education
# Unique Higher Education status
print(student_por.higher.unique())
student_por_clean.loc[student_por_clean['higher']=='yes', 'higher'] = 2
student_por_clean.loc[student_por_clean['higher']=='no', 'higher'] = 1
print(student_por_clean.higher.unique())
print()

# Internet
# Unique Internet Status
print(student_por.internet.unique())
student_por_clean.loc[student_por_clean['internet']=='yes', 'internet'] = 2
student_por_clean.loc[student_por_clean['internet']=='no', 'internet'] = 1
print(student_por_clean.internet.unique())
print()


# Relationship
# Unique Relationship Status
print(student_por.romantic.unique())
student_por_clean.loc[student_por_clean['romantic']=='yes', 'romantic'] = 2
student_por_clean.loc[student_por_clean['romantic']=='no', 'romantic'] = 1
print(student_por_clean.romantic.unique())
print()

"""### 1> Applying SVR (Support Vector Machine Regression)

#### Mathematics Training and Testing Sets
"""

mat_data_training = student_mat_clean.loc[:380, :'G2']
mat_label_training = student_mat_clean.loc[:380, 'G3' :'G3']
mat_label_training = mat_label_training.values.flatten()

mat_data_testing = student_mat_clean.loc[380:, :'G2']
mat_label_testing = student_mat_clean.loc[380:, 'G3' :'G3']

"""#### Portuguese Training and Testing Sets"""

por_data_training = student_por_clean.loc[:380, :'G2']
por_label_training = student_por_clean.loc[:380, 'G3' :'G3']
por_label_training = por_label_training.values.flatten()

por_data_testing = student_por_clean.loc[380:, :'G2']
por_label_testing = student_por_clean.loc[380:, 'G3' :'G3']

"""#### SVR Mathematics"""

# # Defining 1 degree SVM Regression
# clf_1 = SVR(C=0.2, epsilon=0.2, kernel='poly', degree=1)

# # Fitting SVR
# clf_1.fit(mat_data_training, mat_label_training.ravel())

# # Saving SVR Model
# check = pickle.dumps(clf_1)

# # Predicting the list
# clf_1_results = clf_1.predict(mat_data_testing).tolist()


# # Root Mean Square Error
# mse = mean_squared_error(clf_1_results, mat_label_testing)
# mse_scaled = sqrt(mse)
# print(mse_scaled)

# # Predicting the list
# clf_1_results_check = clf_1.predict(mat_data_testing).tolist()
# print(clf_1_results_check)

"""#### SVR Portuguese"""

# # Defining 1 degree SVM Regression
# clf_2 = SVR(C=0.2, epsilon=0.2, kernel='poly', degree=1)

# # Fitting SVR
# clf_2.fit(por_data_training, por_label_training)

# # Saving SVR Model
# check = pickle.dumps(clf_2)

# # Predicting the list
# clf_2_results = clf_2.predict(por_data_testing).tolist()

# mse = mean_squared_error(clf_2_results, por_label_testing)
# mse_scaled = sqrt(mse)
# print(mse_scaled)

"""### 2> Applying Random forest for prediction"""



"""#### For Mathematics"""

import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

# Function to calculate Mean Squared Error
def calculate_mse(y):
    mean_y = np.mean(y)
    return np.mean((y - mean_y) ** 2)

# Function to create bootstrap samples
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
    return X[indices], y[indices]

# Custom Decision Tree Regressor
class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split or np.unique(y).size == 1:
            return np.mean(y)  # Return mean value for regression

        # Initialize variables for best split
        best_feature, best_threshold, best_mse = None, None, float('inf')
        n_samples, n_features = X.shape

        # Iterate over features and thresholds
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if not np.any(left_indices) or not np.any(right_indices):
                    continue

                left_y, right_y = y[left_indices], y[right_indices]
                mse = (len(left_y) * calculate_mse(left_y) + len(right_y) * calculate_mse(right_y)) / n_samples

                if mse < best_mse:
                    best_feature, best_threshold, best_mse = feature, threshold, mse

        # If no valid split found, return the mean value
        if best_feature is None:
            return np.mean(y)

        # Recursively build the tree
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_tree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_tree = self.fit(X[right_indices], y[right_indices], depth + 1)

        # Return a dictionary representing the tree node
        self.tree = {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
        }
        return self.tree

    def predict_sample(self, sample, tree):
        # Recursively traverse the tree
        if not isinstance(tree, dict):
            return tree  # Return leaf value
        if sample[tree["feature"]] <= tree["threshold"]:
            return self.predict_sample(sample, tree["left"])
        else:
            return self.predict_sample(sample, tree["right"])

    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])

# Random Forest Implementation with Custom Decision Tree
class RandomForestRegressorScratch:
    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        if random_state:
            np.random.seed(random_state)

    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features  # Use all features

    def fit(self, X, y):
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)
        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            X_sample, y_sample = bootstrap_sample(X, y)

            # Select random subset of features
            feature_idx = np.random.choice(range(n_features), size=max_features, replace=False)
            self.feature_indices.append(feature_idx)

            # Train a custom Decision Tree on the subset
            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_idx], y_sample)
            self.trees.append((tree, feature_idx))

    def predict(self, X):
        # Aggregate predictions from all trees
        predictions = np.array([
            tree.predict(X[:, feature_idx]) for tree, feature_idx in self.trees
        ])
        return np.mean(predictions, axis=0)

# Train Random Forest from scratch
X_train = mat_data_training.values
y_train = mat_label_training
X_test = mat_data_testing.values
y_test = mat_label_testing.values.flatten()

rf_scratch = RandomForestRegressorScratch(
    n_estimators=150,  # Number of trees
    max_features=".8",  # Use sqrt(number of features) for each tree
    max_depth=15,         # Maximum depth of trees
    min_samples_split=5,  # Minimum samples required to split
    random_state=42
)
rf_scratch.fit(X_train, y_train)

# Save the model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_scratch, file)

# Predictions
y_pred = rf_scratch.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
mse_scaled = sqrt(mse)
print("MSE (Scaled):", mse_scaled)

from sklearn.ensemble import RandomForestRegressor
rf_model_mat = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_mat.fit(mat_data_training, mat_label_training)
mat_predict_rf = rf_model_mat.predict(mat_data_testing)

mse2 = mean_squared_error(mat_predict_rf, mat_label_testing)
mse2_scaled = sqrt(mse2)
print(mse2_scaled)

"""#### For Portuguese"""

rf_model_por = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_por.fit(por_data_training, por_label_training)
por_predict_rf = rf_model_por.predict(por_data_testing)

mse2 = mean_squared_error(por_predict_rf, por_label_testing)
mse2_scaled = sqrt(mse2)
print(mse2_scaled)

def get_input():
    # Ask for user input (you can customize these questions based on the dataset)
    school = int(input("Enter the school (1 for GP, 2 for MS): "))
    sex = int(input("Enter gender (1 for M, 2 for F): "))
    address = int(input("Enter address (1 for R, 2 for U): "))
    famsize = int(input("Enter family size (1 for LE3, 2 for GT3): "))
    Pstatus = int(input("Enter parent's status (1 for A, 2 for T): "))
    Mjob = int(input("Enter mother's job (1 for at_home, 2 for other, 3 for health, 4 for services, 5 for teacher): "))
    Fjob = int(input("Enter father's job (1 for at_home, 2 for other, 3 for health, 4 for services, 5 for teacher): "))
    reason = int(input("Enter reason (1 for other, 2 for home, 3 for course, 4 for reputation): "))
    guardian = int(input("Enter guardian (1 for other, 2 for mother, 3 for father): "))
    schoolsup = int(input("Enter school support (1 for no, 2 for yes): "))
    famsup = int(input("Enter family support (1 for no, 2 for yes): "))
    paid = int(input("Enter paid (1 for no, 2 for yes): "))
    activities = int(input("Enter activities (1 for no, 2 for yes): "))
    nursery = int(input("Enter nursery (1 for no, 2 for yes): "))
    higher = int(input("Enter higher education (1 for no, 2 for yes): "))
    internet = int(input("Enter internet access (1 for no, 2 for yes): "))
    romantic = int(input("Enter romantic relationship (1 for no, 2 for yes): "))
    G2 = float(input("Enter G2 grade: "))  # Enter the grade in G2

    # Create a list or array with the input data
    input_data = [[school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup,
                   paid, activities, nursery, higher, internet, romantic, G2]]

    return input_data

import builtins

import builtins

def get_input():
    # Ask for user input (you can customize these questions based on the dataset)
    school = int(builtins.input("Enter the school (1 for GP, 2 for MS): "))
    sex = int(builtins.input("Enter gender (1 for M, 2 for F): "))
    age = int(builtins.input("Enter age: "))
    address = int(builtins.input("Enter address (1 for R, 2 for U): "))
    famsize = int(builtins.input("Enter family size (1 for LE3, 2 for GT3): "))
    Pstatus = int(builtins.input("Enter parent's status (1 for A, 2 for T): "))
    Medu = int(builtins.input("Enter mother's education (0 to 4): "))
    Fedu = int(builtins.input("Enter father's education (0 to 4): "))
    Mjob = int(builtins.input("Enter mother's job (1 for at_home, 2 for other, 3 for health, 4 for services, 5 for teacher): "))
    Fjob = int(builtins.input("Enter father's job (1 for at_home, 2 for other, 3 for health, 4 for services, 5 for teacher): "))
    reason = int(builtins.input("Enter reason (1 for other, 2 for home, 3 for course, 4 for reputation): "))
    guardian = int(builtins.input("Enter guardian (1 for other, 2 for mother, 3 for father): "))
    traveltime = int(builtins.input("Enter travel time (1 to 4): "))
    studytime = int(builtins.input("Enter study time (1 to 4): "))
    failures = int(builtins.input("Enter number of past class failures (0 to 3): "))
    schoolsup = int(builtins.input("Enter school support (1 for no, 2 for yes): "))
    famsup = int(builtins.input("Enter family support (1 for no, 2 for yes): "))
    paid = int(builtins.input("Enter paid (1 for no, 2 for yes): "))
    activities = int(builtins.input("Enter activities (1 for no, 2 for yes): "))
    nursery = int(builtins.input("Enter nursery (1 for no, 2 for yes): "))
    higher = int(builtins.input("Enter higher education (1 for no, 2 for yes): "))
    internet = int(builtins.input("Enter internet access (1 for no, 2 for yes): "))
    romantic = int(builtins.input("Enter romantic relationship (1 for no, 2 for yes): "))
    famrel = int(builtins.input("Enter quality of family relationships (1 to 5): "))
    freetime = int(builtins.input("Enter free time after school (1 to 5): "))
    goout = int(builtins.input("Enter going out with friends (1 to 5): "))
    Dalc = int(builtins.input("Enter workday alcohol consumption (1 to 5): "))
    Walc = int(builtins.input("Enter weekend alcohol consumption (1 to 5): "))
    health = int(builtins.input("Enter current health status (1 to 5): "))
    absences = int(builtins.input("Enter number of school absences (0 to 93): "))
    G1 = float(builtins.input("Enter G1 grade: "))
    G2 = float(builtins.input("Enter G2 grade: "))

    # Create a list or array with the input data
    input_data = [[school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime,
                   studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
                   famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2]]

    return input_data

def predict_performance(input_data):

    # Make a prediction using the trained SVR model (assuming clf_1 is already trained)
    prediction = rf_model_mat.predict(input_data)

    # Output the predicted grade (G3)
    print(f"The predicted grade (G3) for the student is: {prediction[0]}")
    prediction = rf_model_por.predict(input_data)

    # Output the predicted grade (G3)
    print(f"The predicted grade (G3) for the student is: {prediction[0]}")

# In your main execution cell
user_input = get_input() # Changed variable name to 'user_input'
predict_performance(user_input) # Pass 'user_input' to the function
