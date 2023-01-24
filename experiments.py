
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs
# ...
#
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

N = 40
M = 10
iterations = 3

# Create fake data for 4 cases
# Used them from usage.py


def create_dataset(N, M, case):
    if(case == 0):  # Continuous X, Continuous y
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    elif(case == 1):  # Continuous X, Categorical y
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(5, size=N), dtype="category")
    elif(case == 2):  # Categorical X, Categorical y
        X = pd.DataFrame({i: pd.Series(np.random.randint(
            2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(5, size=N), dtype="category")
    else:  # Categorical X, Continuous y
        X = pd.DataFrame({i: pd.Series(np.random.randint(
            2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
    return X, y


def time_experiment(N, M):
    # There can be 4 cases: Continuous X, Continuous y, Continuous X, Categorical y, Categorical X, Categorical y, Categorical X, Continuous y
    # case will be used as a parameter in create_dataset() to create the dataset for each case
    for case in range(0, 4):
        avgLearningTime = np.arange(N*M).reshape(N, M).astype(float)
        avgPredictionTime = np.arange(N*M).reshape(N, M).astype(float)
        stdLearningTime = np.arange(N*M).reshape(N, M).astype(float)
        stdPredictionTime = np.arange(N*M).reshape(N, M).astype(float)
        Xaxis = np.arange(1, N+1)
        Yaxis = np.arange(1, M+1)
        for n in range(1, N+1):
            for m in range(1, M+1):
                # temp arrays to store the learning time and prediction time for each iteration
                temp_learningTime = np.zeros(iterations)
                temp_predictionTime = np.zeros(iterations)
                for i in iterations:
                    X, y = create_dataset(n, m, case)
                    model = DecisionTree()
                    # storing the learning time for each iteration
                    start = time.time()
                    model.fit(X, y)
                    end = time.time()
                    temp_learningTime[i] = end-start
                    # storing the prediction time for each iteration
                    start = time.time()
                    y_hat = model.predict(X)
                    end = time.time()
                    temp_predictionTime[i] = end-start
                # calculating the average learning time and prediction time for each N and M
                avgLearningTime[n-1][m-1] = np.mean(temp_learningTime)
                avgPredictionTime[n-1][m-1] = np.mean(temp_predictionTime)
                # calculating the standard deviation of learning time and prediction time for each N and M
                stdLearningTime[n-1][m-1] = np.std(temp_learningTime)
                stdPredictionTime[n-1][m-1] = np.std(temp_predictionTime)
        if(case == 0):
            heading = "Continuous X, Continuous y"
        elif(case == 1):
            heading = "Continuous X, Categorical y"
        elif(case == 2):
            heading = "Categorical X, Categorical y"
        else:
            heading = "Categorical X, Continuous y"

        # Heatmap for average learning time using matplotlib, adding the axis labels and titles and finally displaying the plot
        plt.imshow(avgLearningTime, extent=[
                   Xaxis.min(), Xaxis.max(), Yaxis.min(), Yaxis.max()], origin='lower')
        plt.colorbar(label="Time in seconds")
        plt.xlabel('N')
        plt.ylabel('M')
        plt.title(heading + " Average Learning Time")
        plt.show()

        # Heatmap for standard deviation of learning time using matplotlib, adding the axis labels and titles and finally displaying the plot
        plt.imshow(stdLearningTime, extent=[
                   Xaxis.min(), Xaxis.max(), Yaxis.min(), Yaxis.max()], origin='lower')
        plt.colorbar(label="Time in seconds")
        plt.xlabel('N')
        plt.ylabel('M')
        plt.title(heading + " Standard Deviation: Learning Time")
        plt.show()

        # Heatmap for average prediction time using matplotlib, adding the axis labels and titles and finally displaying the plot
        plt.imshow(avgPredictionTime, extent=[
                   Xaxis.min(), Xaxis.max(), Yaxis.min(), Yaxis.max()], origin='lower')
        plt.colorbar(label="Time in seconds")
        plt.xlabel('N')
        plt.ylabel('M')
        plt.title(heading + " Average Prediction Time")
        plt.show()

        # Heatmap for standard deviation of prediction time using matplotlib, adding the axis labels and titles and finally displaying the plot
        plt.imshow(stdPredictionTime, extent=[
                   Xaxis.min(), Xaxis.max(), Yaxis.min(), Yaxis.max()], origin='lower')
        plt.colorbar(label="Time in seconds")
        plt.xlabel('N')
        plt.ylabel('M')
        plt.title(heading + " Standard Deviation: Prediction Time")
        plt.show()


time_experiment(N, M)
