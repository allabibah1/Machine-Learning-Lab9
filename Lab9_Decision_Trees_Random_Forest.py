# ARTI308 Machine Learning - Lab 9
# Decision Trees and Random Forest Project
# Keep loan_data.csv in the same folder before running.

# Import Libraries  Import the usual libraries for pandas and plotting. You can import sklearn later on.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the Data   Use pandas to read loan_data.csv as a dataframe called loans.

loans = pd.read_csv('loan_data.csv')

loans.info()

loans.head()

loans.describe()

plt.figure(figsize=(10, 6))
loans[loans['credit.policy'] == 1]['fico'].hist(bins=30, alpha=0.5, label='credit.policy = 1')
loans[loans['credit.policy'] == 0]['fico'].hist(bins=30, alpha=0.5, label='credit.policy = 0')
plt.xlabel('FICO Score')
plt.legend()
plt.show()

# Create a similar figure, except this time select by the not.fully.paid column.

plt.figure(figsize=(10, 6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(bins=30, alpha=0.5, label='not.fully.paid = 1')
loans[loans['not.fully.paid'] == 0]['fico'].hist(bins=30, alpha=0.5, label='not.fully.paid = 0')
plt.xlabel('FICO Score')
plt.legend()
plt.show()

# Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.

plt.figure(figsize=(11, 5))
sns.countplot(data=loans, x='purpose', hue='not.fully.paid')
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

# Let's see the trend between FICO score and interest rate. Recreate the following jointplot.

sns.jointplot(data=loans, x='fico', y='int.rate', color='purple')
plt.show()

sns.lmplot(data=loans, x='fico', y='int.rate', hue='credit.policy',
           col='not.fully.paid', palette='Set1', height=5,
           scatter_kws={'alpha': 0.3}, ci=None)
plt.show()

# Setting up the Data  Let's get ready to set up our data for our Random Forest Classification Model!  Check loans.info() again.

loans.info()

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

final_data.head()

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Training a Decision Tree Model  Let's start by training a single decision tree first!   Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

# Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.

dtree = DecisionTreeClassifier(random_state=101)

dtree.fit(X_train, y_train)

# Predictions and Evaluation of Decision Tree Create predictions from the test set and create a classification report and a confusion matrix.

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=101)

rfc.fit(X_train, y_train)

# Predictions and Evaluation  Let's predict off the y_test values and evaluate our model.   Predict the class of not.fully.paid for the X_test data.

rfc_pred = rfc.predict(X_test)

# Now create a classification report from the results. Do you get anything strange or some sort of warning?

print(classification_report(y_test, rfc_pred))

print('Decision Tree accuracy:', round(accuracy_score(y_test, predictions), 4))
print('Random Forest accuracy:', round(accuracy_score(y_test, rfc_pred), 4))

# Show the Confusion Matrix for the predictions.

print(confusion_matrix(y_test, rfc_pred))

# What performed better the random forest or the decision tree?
