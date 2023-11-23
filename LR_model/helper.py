import csv
import pandas as pd
import numpy as np
import warnings
import math

#suppress warnings
warnings.filterwarnings('ignore')

def parseData(filename):
    csv_data = pd.read_csv(filename)
    numpy_data = csv_data.values
    rows, columns = numpy_data.shape
    X = numpy_data[:, :columns - 1]
    y = numpy_data[:, columns - 1:]
    X = np.array(X)
    y = np.array(y)
    return X, y

def parseDataBreastCancer(filename):
    csv_data = pd.read_csv(filename)
    numpy_data = csv_data.values
    rows, columns = numpy_data.shape
    X = numpy_data[:, 2:]
    y = numpy_data[:, 1:2]
    X = np.array(X)
    y = np.array(y)
    return X, y

def parseDataSpamEmail(filename):
    csv_data = pd.read_csv(filename)
    numpy_data = csv_data.values
    rows, columns = numpy_data.shape
    X = numpy_data[:, 4:]
    y = numpy_data[:, columns - 1:]
    X = np.array(X)
    y = np.array(y)

    date_idx = 0
    time_idx = 1

    for x in X:
        date = x[date_idx]
        date = date.replace("-", "")
        x[date_idx] = int(date)
        time = x[time_idx]
        time = time.replace(":", "")
        x[time_idx] = int(time)

    return X, y

def reshape_y(y):
    ''' 
    y shape is 2d array of array.
    '''
    y_new = np.array([lst[0] for lst in y])
    return y_new

def splitData(X, y, trainSplit, valSplit, testSplit):
    trainStop = int(trainSplit * X.shape[0])
    valStop = int((trainSplit + valSplit) * X.shape[0])
    train_x = X[0:trainStop, :]
    train_y = y[0:trainStop]
    val_x = X[trainStop:valStop, :]
    val_y = y[trainStop:valStop]
    test_x = X[valStop:, :]
    test_y = y[valStop:]
    return train_x, reshape_y(train_y), val_x, reshape_y(val_y), test_x, reshape_y(test_y)

def splitData2(X, y, trainSplit, valSplit, testSplit):
    trainStop = int(trainSplit * X.shape[0])
    valStop = int((trainSplit + valSplit) * X.shape[0])
    train_x = X[0:trainStop, :]
    train_y = y[0:trainStop]
    val_x = X[trainStop:valStop, :]
    val_y = y[trainStop:valStop]
    test_x = X[valStop:, :]
    test_y = y[valStop:]
    return train_x, train_y, val_x, val_y, test_x, test_y


def normalize(X):
    rangeX = np.zeros(X.shape[1])
    minX = np.zeros(X.shape[1])
    normX = np.zeros(X.shape)

    for i in range(X.shape[1]):
        minX[i] = min(X[:, i])
        rangeX[i] = max(X[:, i]) - minX[i]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            normX[i][j] = (X[i][j] - minX[j]) / rangeX[j]

    for i in range(len(normX)):
        for j in range(len(normX[i])):
            if math.isnan(normX[i][j]):
                normX[i][j] = 0

    return normX

def normalizeBreastCancer(X):
    print(X.shape)

    rangeX = np.zeros(X.shape[1])
    minX = np.zeros(X.shape[1])
    normX = np.zeros(X.shape)


    for i in range(1,X.shape[1]):
        minX[i] = min(X[:, i])
        rangeX[i] = max(X[:, i]) - minX[i]
    for i in range(1,X.shape[0]):
        for j in range(1, X.shape[1]):
            normX[i][j] = (X[i][j] - minX[j]) / rangeX[j]

    for i in range(X.shape[1]):
        normX[i][0] = 1.00 if X[i][0] == "M" else 0

    return normX

def score(predictions, y):
    numCorrect = np.sum(predictions == y.T)
    accuracy = numCorrect / y.shape[0]
    return accuracy