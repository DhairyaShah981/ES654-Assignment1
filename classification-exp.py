from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read dataset
# ...
#

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Train-test split (70-30)
X_train = pd.DataFrame(X[0:70, :])
y_train = pd.Series(y[0:70], dtype="category")
X_test = pd.DataFrame(X[70:100, :])
y_test = pd.Series(y[70:100], dtype="category")

#  2(a) Real input and discrete output with 70-30 split
for criteria in ['information_gain', 'gini_index']:
    model = DecisionTree(criterion=criteria)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    model.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))


# 2 (b) 5 fold cross validation
# function to find optimal depth and criteria (hyperparameter tuning)


def kfolds_opt_depth(X, y, outFolds=5, inFolds=4, depths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    # Converting into dataframes and series
    X = pd.DataFrame(X)
    y = pd.Series(y)
    outFolds = 5  # number of folds for train-test
    inFolds = 4  # number of folds for validation-train
    ind = 20  # (total/number of folds)
    hyperparams = []
    for i in range(outFolds):
        X_test = pd.DataFrame(X[i*ind:(i+1)*ind]).reset_index(drop=True)
        y_test = pd.Series(y[i*ind:(i+1)*ind]).reset_index(drop=True)
        X_mainTrain = pd.concat(
            (X[0:i*ind], X[(i+1)*ind:]), axis=0).reset_index(drop=True)
        y_mainTrain = pd.concat(
            (y[0:i*ind], y[(i+1)*ind:]), axis=0).reset_index(drop=True)
        optAvgValidationAccuracy = None
        final_depth = None
        final_criteria = None

        for criteria in ['information_gain', 'gini_index']:
            for depth in depths:
                AvgValidationAccuracy = 0
                for j in range(inFolds):
                    X_actualTrain = pd.concat(
                        (X_mainTrain[0:j*ind], X_mainTrain[(j+1)*ind:]), axis=0).reset_index(drop=True)
                    y_actualTrain = pd.concat(
                        (y_mainTrain[0:j*ind], y_mainTrain[(j+1)*ind:]), axis=0).reset_index(drop=True)
                    X_validation = pd.DataFrame(
                        X_mainTrain[j*ind:(j+1)*ind]).reset_index(drop=True)
                    y_validation = pd.Series(
                        y_mainTrain[j*ind:(j+1)*ind]).reset_index(drop=True)
                    model = DecisionTree(criterion=criteria, max_depth=depth)
                    model.fit(X_actualTrain, y_actualTrain)
                    y_hat = model.predict(X_validation)
                    AvgValidationAccuracy += accuracy(y_hat, y_validation)
                AvgValidationAccuracy /= inFolds
                if optAvgValidationAccuracy is None or AvgValidationAccuracy > optAvgValidationAccuracy:
                    optAvgValidationAccuracy = AvgValidationAccuracy
                    final_depth = depth
                    final_criteria = criteria
        final_model = DecisionTree(
            criterion=final_criteria, max_depth=final_depth)
        final_model.fit(X_mainTrain, y_mainTrain)
        y_hat = final_model.predict(X_test)
        hyperparams.append(
            (final_depth, final_criteria, accuracy(y_hat, y_test)))
    return hyperparams


ans = kfolds_opt_depth(X, y, outFolds=5, inFolds=4,
                       depths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
summed_accuracy = 0
for i in range(5):
    summed_accuracy += ans[i][2]
avg_accuracy_kfolds = float(summed_accuracy)/5
# It prints the optimal depth and criteria, and the corresponding accuracy for each fold
print(ans)
# Print the average accuracy over all 5 models
print(avg_accuracy_kfolds)
