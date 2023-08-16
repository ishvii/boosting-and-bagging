# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('dataset.csv')

# Preprocess the data (e.g., handle missing values, encode categorical variables)

# Split the data into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the base learner (e.g., Decision Tree Classifier)
base_learner = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost Classifier with 100 estimators
adaboost_classifier = AdaBoostClassifier(base_estimator=base_learner, n_estimators=100, random_state=42)

# Train the model on the training data
adaboost_classifier.fit(X_train, y_train)

# Make predictions on the testing data
adaboost_predictions = adaboost_classifier.predict(X_test)

# Calculate the accuracy of the model
adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)

# Print the accuracy
print("AdaBoost Accuracy:", adaboost_accuracy)
