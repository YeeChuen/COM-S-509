import csv
import numpy as np 
import matplotlib.pyplot as plt
import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot
import numpy.linalg 
import numpy.random
from sklearn.utils import shuffle



#data parsing from csv to samples and labels
#called by specific parsing methods such as "parseAlzheimers"
# filename - name of file
#returns list of samples X, list of labels y
def parseData(filename):
    csv_data = pd.read_csv(filename)
    numpy_data = csv_data.values
    rows, columns = numpy_data.shape
    X = numpy_data[:, :columns - 1]
    y = numpy_data[:, columns - 1:]
    X = np.array(X)
    y = np.array(y)
    return X, y


#parsing method for specific file for readability
# filename - name of the file
#returns list of samples X, and list of labels y
def parseAlzheimers(filename):
    X, y = parseData("handwriting_alzheimers.csv")
    X = X[:, 1:]
    y = np.where(y == "P", 1, y)
    y = np.where(y == "H", -1, y)

    return X, y


#splits given data based on the partitions given
# X - samples
# y - labels
# trainSplit - float from 0.0 to 1.0 dictating % of training data
# valSplit - float from 0.0 to 1.0, same as trainSplit but for validation data
# testSplit - float from 0.0 to 1.0, same as trainSplit but for testing data
#returns samples and labels for training, validation and testing
def splitData(X, y, trainSplit, valSplit, testSplit):
    trainStop = int(trainSplit * X.shape[0])
    valStop = int((trainSplit + valSplit) * X.shape[0])
    train_x = X[0:trainStop, :]
    train_y = y[0:trainStop]
    val_x = X[trainStop:valStop, :]
    val_y = y[trainStop:valStop]
    test_x = X[valStop:, :]
    test_y = y[valStop:]
    return train_x, train_y, val_x, val_y, test_x, test_y


#data partition method to split data for different weak models in bagging
#this is sampling with replacement
# tX - training samples
# tY - training labels
# splits - number of models that will be used in bagging
# spp - samples per partition; int
#returns list of lists of samples and labels for training and validation
def bagging_split(tX, tY, splits, spp):
    tXs = []
    tYs = []

    size = tX.shape[0]

    for i in range(splits):
        tx = []
        ty = []
        
        for j in range(spp):
            index = np.random.randint(size)
            tx.append(tX[index])
            ty.append(tY[index])
            
        tx = np.array(tx)
        ty = np.array(ty)
        tXs.append(tx)
        tYs.append(ty)

    return tXs, tYs


#data partition for boosting
#sampling with replacement
# tX - training samples
# tY - training labels
# spp - samples per partition; int
#returns list of samples, list of labels, and list of index locations of samples
def boosting_split(tX, tY, spp):
    tx = []
    ty = []
    ti = []

    size = tX.shape[0]

    for i in range(spp):
        index = np.random.randint(size)
        tx.append(tX[index])
        ty.append(tY[index])
        ti.append(index)

    tx = np.array(tx)
    ty = np.array(ty)
    ti = np.array(ti)

    return tx, ty, ti


#normalization method over a range of 1
# X - samples to be normalized
#returns normalized samples X
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

    return normX


#loss function for SVM; hinge-loss
# C - coefficient factor for regularizing; weights if boosting
# w - weights for the model
# b - bias for the model
# X - samples
# y - labels
#returns a list of losses
def hingeLoss(C, w, b, X, y):
    reg_term = 0.5 * (w.T @ w)
    losses = np.zeros(X.shape[0])

    noboost = False
    if (isinstance(C, int) or isinstance(C, float)):
        noboost = True

    for i in range(X.shape[0]):
        opt_term = y[i] * ((w.T @ X[i]) + b)
        if (noboost == True):
            losses[i] = reg_term + C * max(0, 1 - opt_term)
        else:
            losses[i] = reg_term + C[i] * max(0, 1 - opt_term)
    
    return losses


#training method for SVM model
# C - coefficient factor used in loss function
# X - training samples
# y - training labels
# batchSize - hyperparameter
# learningRate - hyperparameter
# epochs - hyperparameter
#returns the model consisting of weights, bias, and a list of losses
def fit(C, X, y, batchSize, learningRate, epochs):
    w = np.zeros(X.shape[1])
    b = 0
    lossList = []

    for i in range(epochs):
        losses = hingeLoss(C, w, b, X, y)
        lossList.append(losses)
        for batch in range(0, X.shape[0], batchSize):
            batchX = X[batch:batch+batchSize, :]
            batchy = y[batch:batch+batchSize, :]
            wGradient, bGradient = gradient(C, batchX, batchy, w, b)
            w = w - learningRate * wGradient
            b = b - learningRate * bGradient
    return w, b, lossList


#computes the gradient of a particular iteration
# C - coefficient for regularization; weights if boosting
# X - training samples
# y - training labels
# w - current weights
# b - current bias
#returns the gradient for weights and bias separately
def gradient(C, X, y, w, b):
    wGradient = 0
    bGradient = 0

    noboost = False
    if (isinstance(C, int) or isinstance(C, float)):
        noboost = True

    for i in range(X.shape[0]):
        dist = y[i] * (w.T @ X[i] + b)

        if (dist < 1):
            if (noboost == True):
                wGradient += -1 * (C * y[i] * X[i])
                bGradient += -1 * (C * y[i])
            else:
                wGradient += -1 * (C[i] * y[i] * X[i])
                bGradient += -1 * (C[i] * y[i])
    return wGradient, bGradient


#prediction method for single model
# X - samples to be tested on (usually testing samples)
# w - weights of the model
# b - bias of the model
#returns a list of predictions for the given model on X
def predict(X, w, b):
    predictions = w @ X.T + b
    predictions = np.sign(predictions)
    return predictions


#scoring method for some given predictions
# predictions - list of predictions made on X with some model
# y - true labels of the samples
#returns a float between 0.0 and 1.0 denoting the accuracy of the predictions
def score(predictions, y):
    numCorrect = np.sum(predictions == y.T)
    accuracy = numCorrect / y.shape[0]
    return accuracy


#polynomial transformation
# X - input matrix to be transformed
# g - gamma coefficient modifying the dot product
# c - constant added to the dot product result
# d - polynomial degree
#returns transformed matrix X
def poly_kernel(X, g, c, d):
    n = X.shape[0]
    Xk = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Xk[i, j] = (g * X[i] @ X[j].T + c) ** d

    return Xk


#radial basis function transformation
# X - input matrix to be transformed
# g - gamma coefficient modifier, this value is multiplied by -1
#returns transformed matrix X
def rbf_kernel(X, g):
    n = X.shape[0]
    Xk = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            norm_sq = np.linalg.norm(X[i] - X[j]) ** 2
            Xk[i, j] = np.exp(-g * norm_sq)
    
    return Xk


#sigmoid kernel transformation
# X - input matrix to be transformed
# g - gamma coefficient modifier
# c- constant added to dot product
#returns transformed matrix X
def sigmoid_kernel(X, g, c):
    n = X.shape[0]
    Xk = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Xk[i, j] = np.tanh(g * (X[i] @ X[j]) + c)
    
    return Xk


#ensemble learning - bagging
# tX - training samples
# tY - training labels
# vX - validation samples
# vY - validation labels
# batchSize - hyperparameter
# learningRate - hyperparameter
# epochs - hyperparameter
# numModels - dictates how many weak classifiers are trained
# spp - samples per partition for each model; int
#returns list of weights, list of biases, and list of losses
#sanity; make sure numModels <= vX.size
def bagging(tX, tY, vX, vY, C, batchSize, learningRate, epochs, numModels, spp):
    ws = []
    bs = []
    ls = []

    tXs, tYs = bagging_split(tX, tY, numModels, spp)

    for i in range(numModels):
        w, b, lossList = fit(C, tXs[i], tYs[i], batchSize, learningRate, epochs)
        ws.append(w)
        bs.append(b)
        ls.append(lossList)

    return ws, bs, ls


#prediction function for ensemble learning
# X - testing samples
# ws - list of weights; 2d
# bs - list of biases; 2d
# weighted - boolean variable; 
#    if true, accuracy of model will impact the weight of its vote
# accs - accuracy of models
# offset - float 0 <= offset < 1;
#    advanced optimization, offsets model contribution to prediction by
#    the set amount. 0.5 flips the prediction contribution of models with 
#    less than 50% accuracy
#    this parameter also sharply reduces the contribution of models near the offset
#    default = 0
#returns the predictions for all input samples X
def ensemble_predict(X, ws, bs, weighted, accs, offset = 0.0):
    predictions = np.zeros(X.shape[0])

    if (weighted == True):
        accs = np.array(accs)
        accs -= offset
        for i in range(len(ws)):
            preds = ws[i] @ X.T + bs[i]
            preds = np.float64(preds)
            preds *= accs[i]
            predictions += preds
        
        predictions = np.sign(predictions)
    else:
        for i in range(len(ws)):
            preds = ws[i] @ X.T + bs[i]
            preds = np.float64(preds)
            preds = np.sign(preds)
            predictions += preds
        
        predictions = np.sign(predictions)

    return predictions


#calculates a list of accuracies for multiple models
#indeed use for bagging and boosting
# X - testing samples
# y - testing labels
# ws - list of weights representing the models; 2d
# bs - list of biases for the models
#returns accuracy of all models
def get_bagging_acc(X, y, ws, bs):
    accs = []

    for i in range(len(ws)):
        predictions = predict(X, ws[i], bs[i])
        accs.append(score(predictions, y))

    return accs


#wrapper for ensemble learning prediction
# X - test samples
# y - test labels
# ws - list of weights
# bs - list of biases
# weighted - boolean 
# offset - float
# accs - model influence on prediction
#returns accuracy of bagged models
def ensemble_wrapper(X, y, ws, bs, weighted, offset = 0.0, accs = 0):
    if (accs == 0):
        accs = get_bagging_acc(X, y, ws, bs)

    predictions = ensemble_predict(X, ws, bs, weighted, accs, offset)
    acc = score(predictions, y)

    return acc


#wrapper for single model predictions
# X - test samples
# y - test labels
# w - weights
# b - bias
#returns accuracy of model
def wrapper(X, y, w, b):
    predictions = predict(X, w, b)
    acc = score(predictions, y)

    return acc


#boosting method
# tX - training samples
# tY - training labels
# vX - validation samples
# vY - validation labels
# batchSize - hyperparameter
# learningRate - hyperparameter
# epochs - hyperparameter
# numModels - number of models used in ensemble
# spp - samples per parition
#returns list of weights, list of biases, list of losses, and list of model contributions
def boosting(tX, tY, vX, vY, batchSize, learningRate, epochs, numModels, spp):
    ws = []
    bs = []
    ls = []
    sWs = np.ones(tX.shape[0])
    sWs /= tX.shape[0]
    lWs = []

    for i in range(numModels):
        tx, ty, ti = boosting_split(tX, tY, spp)
        w, b, lossList = fit(1, tx, ty, batchSize, learningRate, epochs)
        ws.append(w)
        bs.append(b)
        ls.append(lossList)

        predictions = predict(tx, w, b)
        ty = ty.flatten()
        mscInd = np.where(predictions != ty.T)[0]
        error = 0.0
        for j in range(mscInd.shape[0]):
            error +=  sWs[ti[mscInd[j]]]
        
        beta = np.log(((1 - error) / max(error, 1e-10)))
        delta = np.exp(beta)

        for j in range(mscInd.shape[0]):
            sWs[ti[mscInd[j]]] = sWs[ti[mscInd[j]]] * delta

        sWs /= np.sum(sWs)
        lWs.append(beta)

    return ws, bs, ls, lWs


#simulates data corruption by changing feature values to 0
# X - input samples to be corrupted
# dcSamples - number of corrupted samples; float between 0.0 - 1.0
# dcFeatures - number of corrupted features in corrupted samples; 0.0 - 0.99
# uniform - boolean signifying if all corrupted samples have the same number of corrupted features
#returns corrupted samples X
def applyCorruption(X, dcSamples, dcFeatures, uniform):
    numCorS = int(dcSamples * X.shape[0])
    numCorF = int(dcFeatures * X.shape[1])

    if (uniform == True):
        for i in range(numCorS):
            for j in range(numCorF):
                corInd = np.random.randint(X.shape[1])
                X[i, corInd] = 0.0
    else:
        for i in range(numCorS):
            numCorR = np.random.randint(numCorF)
            for j in range(numCorR):
                corInd = np.random.randint(X.shape[1])
                X[i, corInd] = 0.0

    return X


#calculates the correlation coefficient between all features
#formula utilized is "pearson's r"
# X - input samples
#returns a correlation matrix cor of all correlation coefficients
def calc_correlation(X):
    mean = np.zeros(X.shape[1])
    std = np.zeros(X.shape[1])
    cov = np.zeros((X.shape[1], X.shape[1]))
    cor = np.zeros((X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        mean[i] = np.sum(X[:, i]) / X.shape[1]

    for i in range(X.shape[1]):
        std[i] = np.sqrt(np.sum((X[:, i] - mean[i]) ** 2) / X.shape[1])

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            c = 2
            if (i == j):
                c = 1
            cov[i, j] = np.sum((X[:, i] - mean[i]) * (X[:, j] - mean[j])) / (c * X.shape[1])

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            cor[i, j] = cov[i, j] / (std[i] * std[j])

    return cor


#calculates coefficient factor between mean of features
# X - input samples
#returns a coefficient matrix coef
def calc_coefficient(X):
    mean = np.zeros(X.shape[1])
    coef = np.zeros((X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        mean[i] = np.sum(X[:, i]) / X.shape[1]

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            coef[i, j] = mean[i] / mean[j]

    return coef


#data reconstruction on some samples X
# X - input samples
# t - correlation threshold before data reconstruction is skipped; 0.0 <= t <= 1.0
#returns X with reconstructed samples based on parameters
def data_reconstruct(X, t):
    cor = calc_correlation(X)
    coef = calc_coefficient(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            corIndex = -1
            val = 0.0
            corList = cor[j, :]

            while (X[i, j] == 0.0):
                corIndex = np.where(np.absolute(corList) == max(np.absolute(corList)))
                val = X[i, corIndex]
                
                if (corList[corIndex] < t):
                    break

                if (val != 0.0):
                    X[i, j] = X[i, corIndex] * coef[corIndex, j]
                else:
                    corList[corIndex] = 0.0

    return X


if __name__ == "__main__":
    #data parsing and trimming
    X, y = parseAlzheimers("handwriting_alzheimers.csv")

    #important note here that normalization comes before kernelization
    #we can also apply normalization after kernelization or both
    X = normalize(X)

    #due to the data having a low sample count, depending on the distribution of the shuffle
    #accuracy can be extremely poor
    #while shuffling help negate this, we need another technique to increase sample count
    X, y = shuffle(X, y)

    #kernelization happens here, comment out to skip kernelization
    X = poly_kernel(X, 1, 1, 2)
    #X = rbf_kernel(X, 1)
    #X = sigmoid_kernel(X, 1, 1)

    #normalization can happen here after kernelization
    X = normalize(X)

    #data split ratios
    train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.7, 0.15, 0.15)

    #training procedures, use only 1
    #ws, bs, ls = bagging(train_x, train_y, val_x, val_y, 1, 5, 0.001, 100, 10, int(train_x.shape[0] / 3))
    ws, bs, ls, lWs = boosting(train_x, train_y, val_x, val_y, 5, 0.001, 100, 10, int(train_x.shape[0] / 3))
    #w, b, lossList = fit(1, train_x, train_y, 10, 0.001, 100)

    #data corruption simulation for robustness
    test_x = applyCorruption(test_x, 0.1, 0.1, False)

    #data reconstruction for corrupted data
    test_x = data_reconstruct(test_x, 0.25)

    #prediction and scoring, use corresponding methods
    #ensemble predictions; set weight to False for bagging, True for boosting
    acc = ensemble_wrapper(test_x, test_y, ws, bs, True, 0.5, lWs)

    #single model predictions
    #acc = wrapper(test_x, test_y, w, b)

    print(acc)