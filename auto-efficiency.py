import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor
from metrics import *


np.random.seed(42)

# Read real-estate data set
# ...
#

k = 8  # number of folds

# We have deleted the column containing the car names as they are unique for each instance
# We did this in the csv file itself
df2 = pd.read_csv('auto-mpg.csv')

# There are some ? iN the horsepower column of data set so we simply drop them
df2 = df2[df2['horsepower'] != '?'].reset_index(drop=True)

# We rename the columns with 0 based indexing to use our DecisionTree classifier
df2.rename(columns={'mpg': 0}, inplace=True)
df2.rename(columns={'cylinders': 1}, inplace=True)
df2.rename(columns={'displacement': 2}, inplace=True)
df2.rename(columns={'horsepower': 3}, inplace=True)
df2.rename(columns={'weight': 4}, inplace=True)
df2.rename(columns={'acceleration': 5}, inplace=True)
df2.rename(columns={'model year': 6}, inplace=True)
df2.rename(columns={'origin': 7}, inplace=True)


# Splitting the data into features and labels
y = df2[0]
X = df2.drop(0, axis=1)

# After splitting again shifting the indices to 0 based indexing
X.rename(columns={1: 0}, inplace=True)
X.rename(columns={2: 1}, inplace=True)
X.rename(columns={3: 2}, inplace=True)
X.rename(columns={4: 3}, inplace=True)
X.rename(columns={5: 4}, inplace=True)
X.rename(columns={6: 5}, inplace=True)
X.rename(columns={7: 6}, inplace=True)

# Train-test split
X_mainTrain = X[:320].reset_index(drop=True)
y_mainTrain = y[:320].reset_index(drop=True)
X_test = X[320:].reset_index(drop=True)
y_test = y[320:].reset_index(drop=True)

# Converting into series and dataframes
X_mainTrain = pd.DataFrame(X_mainTrain, dtype=np.float64)
y_mainTrain = pd.Series(y_mainTrain, dtype=np.float64, name=None)
X_test = pd.DataFrame(X_test, dtype=np.float64)
y_test = pd.Series(y_test, dtype=np.float64, name=None)

# Hyperparams
ind = int(X_mainTrain.shape[0]/k)
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
optDepth = None
optAccuracy = None
minmError = np.inf

# Finding the optimal depth
for depth in depths:
    summed_error = 0
    for i in range(k):
        X_train = pd.concat(
            (X_mainTrain[0:i*ind], X_mainTrain[(i+1)*ind:]), axis=0).reset_index(drop=True)
        y_train = pd.concat(
            (y_mainTrain[0:i*ind], y_mainTrain[(i+1)*ind:]), axis=0).reset_index(drop=True)
        X_validation = X_mainTrain[i*ind:(i+1)*ind].reset_index(drop=True)
        y_validation = y_mainTrain[i*ind:(i+1)*ind].reset_index(drop=True)

        model = DecisionTree(max_depth=depth)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_validation)
        summed_error += (rmse(y_hat, y_validation))

    if((summed_error/k) < minmError):
        minmError = summed_error/k
        optDepth = depth

# Training the dataset on our own decision tree
ourmodel = DecisionTree(max_depth=optDepth)
ourmodel.fit(X_mainTrain, y_mainTrain)
y_hat = ourmodel.predict(X_test)
ourRMSE = rmse(y_hat, y_test)

# Training the dataset on sklearn decision tree
sktree = DecisionTreeRegressor(random_state=0, max_depth=optDepth)
sktree.fit(X_mainTrain, y_mainTrain)
y_skhat = sktree.predict(X_test)
sklearnRMSE = rmse(y_skhat, y_test)

# Printing the results
print("Optimal Depth:", optDepth)
print("RMSE for our decision tree:", ourRMSE)
print("RMSE for sklearn decision tree:", sklearnRMSE)
