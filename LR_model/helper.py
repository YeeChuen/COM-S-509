import csv
import pandas as pd
import numpy as np
import warnings
import math

#suppress warnings
warnings.filterwarnings('ignore')


# get X, y for all 4 data sets
# index:    0-alzheimers, 1-breastcancer, 2-spamemail, 3-water-potability
def get_datasets():
    project_path = '/work/ratul1/chuen/temp/COM-S-573'
    hand_writing_csv = f"{project_path}/data/handwriting_alzheimers.csv"
    X_hw, y_hw = parseData(hand_writing_csv, form = 0)

    breast_cancer_csv = f"{project_path}/data/breast-cancer.csv"
    X_bc, y_bc = parseData(breast_cancer_csv, form = 1)

    spam_email_csv = f"{project_path}/data/spam_email_dataset.csv"
    X_se, y_se = parseData(spam_email_csv, form = 2)

    water_potability_csv = f"{project_path}/data/water_potability.csv"
    X_wp, y_wp = parseData(water_potability_csv, form = 3)

    return {'Alzheimers Handwriting': [X_hw, reshape_y(y_hw)],
            'Breast Cancer': [X_bc, reshape_y(y_bc)], 
            'Spam Email': [X_se, reshape_y(y_se)], 
            'Water Potability': [X_wp, reshape_y(y_wp)]}

#condensed data parsing method
#note that invalid values in the dataset are replaced with 0.0
# filename - name of the file
# form - format of the data; 
#     0-alzheimers, 1-breastcancer, 2-spamemail, 3-water-potability
#returns list of samples X, and list of labels y
def parseData(filename, form = -1):
    csv_data = pd.read_csv(filename)
    csv_data = csv_data.fillna(0)
    numpy_data = csv_data.values
    rows, columns = numpy_data.shape
    X = numpy_data
    y = numpy_data

    if (form == 0):
        X = numpy_data[:, 1:columns - 1]
        y = numpy_data[:, columns - 1:]
        y = np.array(y)
        y = np.where(y == "P", 1, y)
        y = np.where(y == "H", -1, y)
    elif (form == 1):
        X = numpy_data[:, 2:]
        y = numpy_data[:, 1:2]
        y = np.array(y)
        y = np.where(y == "M", 1, y)
        y = np.where(y == "B", -1, y)
    elif (form == 2):
        X = numpy_data[:, 6:columns - 1]
        y = numpy_data[:, columns - 1:]
        y = np.array(y)
        y = np.where(y == 0, -1, y)
    elif (form == 3):
        X = numpy_data[:, :columns - 1]
        y = numpy_data[:, columns - 1:]
        y = np.array(y)
        y = np.where(y == 0, -1, y)
    elif (form == -1):
        X = numpy_data[:, :columns - 1]
        y = numpy_data[:, columns - 1:]
        y = np.array(y)

    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = float(X[i][j])

    X = np.array(X)
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
    return train_x, train_y, val_x, val_y, test_x, test_y

def splitData2(X, y, trainSplit, valSplit, testSplit):
    trainStop = int(trainSplit * X.shape[0])
    valStop = int((trainSplit + valSplit) * X.shape[0])
    train_x = X[0:trainStop, :]
    train_y = y[0:trainStop]
    val_x = X[trainStop:valStop, :]
    val_y = y[trainStop:valStop]
    test_x = X[valStop:, :]
    test_y = y[valStop:]
    return train_x, reshape_y(train_y), val_x, reshape_y(val_y), test_x, reshape_y(test_y)

def nan_value(X, nan = 'zero'):
    meanX = np.nanmean(X, axis = 0)
    medianX = np.nanmedian(X, axis = 0)

    for i in range(len(X)):
        for j in range(len(X[i])):
            if math.isnan(X[i][j]):
                if nan == 'zero':
                    X[i][j] = 0
                elif nan == 'mean':
                    X[i][j] = meanX[j]
                elif nan == 'median':
                    X[i][j] = medianX[j]

    return X


def normalize(X, nan = 'median'):
    '''
    nan takes argument = ['mean', 'median', 'zero']
    '''
    rangeX = np.zeros(X.shape[1])
    minX = np.zeros(X.shape[1])
    normX = np.zeros(X.shape)

    for i in range(X.shape[1]):
        minX[i] = min(X[:, i])
        rangeX[i] = max(X[:, i]) - minX[i]
    for i in range(X.shape[0]): # row
        for j in range(X.shape[1]): # column
            normX[i][j] = (X[i][j] - minX[j]) / rangeX[j]

    if nan == 'zero':
        return nan_value(normX, nan = 'zero')
    elif nan == 'mean':
        return nan_value(normX, nan = 'mean')
    elif nan == 'median':
        return nan_value(normX, nan = 'median')
    else:
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

def bagging(tX, tY, vX, vY, C, batchSize, learningRate, epochs, numModels, spp, model = None, model_type = None):
    ws = []
    bs = []
    ls = []

    tXs, tYs = bagging_split(tX, tY, numModels, spp)

    for i in range(numModels):
        if model_type in ["multinomial", "binomial"]:
            LRmodel = model(learningRate, epochs, classifier=model_type)
            LRmodel.fit(tXs[i], tYs[i])
            w, b, lossList = LRmodel.get_params()
        
        else:
            w, b, lossList = fit(C, tXs[i], tYs[i], batchSize, learningRate, epochs)
        ws.append(w)
        bs.append(b)
        ls.append(lossList)

    return ws, bs, ls

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

def boosting(tX, tY, vX, vY, batchSize, learningRate, epochs, numModels, spp, model = None, model_type = None):
    ws = []
    bs = []
    ls = []
    sWs = np.ones(tX.shape[0])
    sWs /= tX.shape[0]
    lWs = []

    for i in range(numModels):
        tx, ty, ti = boosting_split(tX, tY, spp)
        predictions = 0
        
        if model_type in ["multinomial", "binomial"]:
            LRmodel = model(learningRate, epochs, classifier=model_type)
            LRmodel.fit(tx, ty)
            w, b, lossList = LRmodel.get_params()
            predictions = LRmodel.predict_with_param(tx, w, b)
        
        else:
            w, b, lossList = fit(1, tx, ty, batchSize, learningRate, epochs)
            predictions = predict(tx, w, b)

        ws.append(w)
        bs.append(b)
        ls.append(lossList)


        ty = ty.flatten()
        mscInd = np.where(predictions != ty.T)[0]
        error = 0.0
        for j in range(mscInd.shape[0]):
            error +=  sWs[ti[mscInd[j]]]
        
        beta = np.log(((1 - error) / max(error, 1e-7)))
        delta = np.exp(beta)

        for j in range(mscInd.shape[0]):
            sWs[ti[mscInd[j]]] = sWs[ti[mscInd[j]]] * delta

        sWs /= np.sum(sWs)
        lWs.append(beta)

    return ws, bs, ls, lWs

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

def predict(X, w, b):
    predictions = w @ X.T + b
    predictions = np.sign(predictions)
    return predictions

def score(predictions, y):
    numCorrect = np.sum(predictions == y.T)
    accuracy = numCorrect / y.shape[0]
    return accuracy

def poly_kernel(X, g, c, d):
    n = X.shape[0]
    Xk = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Xk[i, j] = (g * (X[i] @ X[j].T) + c) ** d

    return Xk

def rbf_kernel(X, g):
    n = X.shape[0]
    Xk = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            norm_sq = np.linalg.norm(X[i] - X[j]) ** 2
            Xk[i, j] = np.exp(-g * norm_sq)
    
    return Xk

def sigmoid_kernel(X, g, c):
    n = X.shape[0]
    Xk = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Xk[i, j] = np.tanh(g * (X[i] @ X[j]) + c)
    
    return Xk

def ensemble_predict(X, ws, bs, weighted, accs, offset = 0.0, model = None, model_type = None):
    predictions = np.zeros(X.shape[0])
    
    non_fit_model = None
    if model_type in ["multinomial", "binomial"]:
        non_fit_model = model(0, 0, classifier=model_type)

    if (weighted == True):
        accs = np.array(accs)
        accs -= offset
        for i in range(len(ws)):
            if model_type in ["multinomial", "binomial"]:
                preds = non_fit_model.predict_with_param(X, ws[i], bs[i])
                preds = np.float64(preds)
                preds *= accs[i]
                predictions += preds

            else:
                preds = ws[i] @ X.T + bs[i]
                preds = np.float64(preds)
                preds *= accs[i]
                predictions += preds
        
        predictions = np.sign(predictions)
    else:
        for i in range(len(ws)):
            if model_type in ["multinomial", "binomial"]:
                preds = non_fit_model.predict_with_param(X, ws[i], bs[i])
                preds = np.float64(preds)
                preds = np.sign(preds)

                predictions += preds


            else:
                preds = ws[i] @ X.T + bs[i]
                preds = np.float64(preds)
                preds = np.sign(preds)
                predictions += preds
        
        predictions = np.sign(predictions)

    return predictions

def get_bagging_acc(X, y, ws, bs, model = None, model_type = None):
    accs = []
    
    non_fit_model = None
    if model_type in ["multinomial", "binomial"]:
        non_fit_model = model(0, 0, classifier=model_type)

    for i in range(len(ws)):
        if model_type in ["multinomial", "binomial"]:
            predictions = non_fit_model.predict_with_param(X, ws[i], bs[i])
            accs.append(non_fit_model.score_with_pred(predictions, y))

        else:
            predictions = predict(X, ws[i], bs[i])
            accs.append(score(predictions, y))

    return accs

def ensemble_wrapper(X, y, ws, bs, weighted, offset = 0.0, accs = 0, model = None, model_type = None):
    if (accs == 0):
        accs = get_bagging_acc(X, y, ws, bs, model = model, model_type = model_type)

    predictions = ensemble_predict(X, ws, bs, weighted, accs, offset, model = model, model_type = model_type)
    
    if model_type in ["multinomial", "binomial"]:
        non_fit_model = model(0, 0, classifier=model_type)
        acc = non_fit_model.score_with_pred(predictions, y)
        return acc

    else:
        acc = score(predictions, y)
        return acc

def wrapper(X, y, w, b):
    predictions = predict(X, w, b)
    acc = score(predictions, y)

    return acc

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

def calc_coefficient(X):
    mean = np.zeros(X.shape[1])
    coef = np.zeros((X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        mean[i] = np.sum(X[:, i]) / X.shape[1]

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            coef[i, j] = mean[i] / mean[j]

    return coef

def data_reconstruct(X, t):
    cor = calc_correlation(X)
    coef = calc_coefficient(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            corIndex = -1
            val = 0.0
            corList = np.copy(cor[j])
            corList[j] = 0.0

            while (X[i, j] == 0.0):
                corIndex = np.where(np.absolute(corList) == max(np.absolute(corList)))[0][0]
                val = X[i, corIndex]
                
                if (corList[corIndex] <= t):
                    break

                if (val != 0.0):
                    X[i, j] = X[i, corIndex] * coef[corIndex, j]
                else:
                    corList[corIndex] = 0.0

    return X

if __name__ == "__main__":
    project_path = '/work/ratul1/chuen/temp/COM-S-573'
    water_potability_csv = f"{project_path}/data/water_potability.csv"
    
    X, y = parseData(water_potability_csv)
    X = X[:,1:]
    y = reshape_y(y)

    Xzero = normalize(X)
    print(Xzero[:10])
    
    Xmean = normalize(X, nan = 'mean')
    print(Xmean[:10])
    
    Xmedian = normalize(X, nan = 'median')
    print(Xmedian[:10])