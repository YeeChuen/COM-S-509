from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail, splitData2, reshape_y
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import sys
import math
import copy

# testing with sklearn library
from sklearn.decomposition import PCA as skPCA

def PCA(X, k = 2):
    no_data = X.shape[0]
    no_feature = X.shape[1]
    
    mean = np.mean(X, axis=0)
    X_hat = X - mean
    S = np.cov(X_hat.T)
    
    eig_vals, eig_vecs = np.linalg.eig(S) 


    '''
    # Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
    # Not as per lecture
    max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
    signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
    eig_vecs = eig_vecs*signs[np.newaxis,:]
    eig_vecs = eig_vecs.T'''

    eig_pairs_list = [(eig_vals[i], eig_vecs[i]) for i in range(no_feature)]
    eig_pairs_list.sort(key=lambda x: x[0], reverse=True)

    X_new = []
    for i in range(int(k)):
        pairs = eig_pairs_list[i]
        values = pairs[1]
        x_new = []
        for x in X_hat:
            temp = x * values
            value_new = np.sum(temp)
            x_new.append(value_new)
        X_new.append(x_new)
    
    return np.array(X_new).T
    pass

def PCAIntegrity():
    data_x1 = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
    ])
    data_x2 = np.array([
        [19, 63],
        [39, 74],
        [30, 87],
        [30, 23],
        [15, 35],
        [15, 43],
        [15, 32],
        [30, 73],
    ])

    X_pca = PCA(data_x2, k = 1)

    # test sklearn PCA
    pca = skPCA(n_components = 1).fit(data_x2)
    X_pca_sklearn = pca.transform(data_x2) # Apply dimensionality reduction to X.

    print(X_pca)
    print(X_pca_sklearn)

def testing():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for handwriting dataset ---")
    hand_writing_csv = f"{project_path}/data/handwriting_alzheimers.csv"

    X, y = parseData(hand_writing_csv)
    X = X[:, 1:]
    y = np.where(y == "P", 1, y)
    y = np.where(y == "H", -1, y)
    X = normalize(X)
    y = reshape_y(y) # <-- this is needed before running feature selection

    X, y = shuffle(X, y)
    X, y = shuffle(X, y)
    X, y = shuffle(X, y)


    data_x1 = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
    ])
    data_x2 = np.array([
        [19, 63],
        [39, 74],
        [30, 87],
        [30, 23],
        [15, 35],
        [15, 43],
        [15, 32],
        [30, 73],
    ])

    X_pca = PCA(X, k = int(X.shape[1] * 0.2))

    print(X_pca)
    print(X_pca_sklearn)
    
    train_x, train_y, _, _, test_x, test_y = splitData2(X, y, 0.8, 0, 0.2)
    print("\n--- No feature reduction ---")
    print(train_x.shape)
    l_rate, no_iter, best_prob = 0.1, 100, 0.5
    print("scratch multi model hyperparam:",str(l_rate), ", ", str(no_iter), ", ", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch multi model prediction: {score}")
    l_rate, no_iter, best_prob = 0.1, 100, 0.5
    print("scratch model hyperparam:",str(l_rate), ", ", str(no_iter), ", ", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction: {score}")
    
    train_x, train_y, _, _, test_x, test_y = splitData2(X_pca, y, 0.8, 0, 0.2)
    print("\n--- PCA feature reduction ---")
    print(train_x.shape)
    l_rate, no_iter, best_prob = 0.1, 100, 0.5
    print("scratch multi model hyperparam:",str(l_rate), ", ", str(no_iter), ", ", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch multi model prediction: {score}")
    l_rate, no_iter, best_prob = 0.1, 100, 0.5
    print("scratch model hyperparam:",str(l_rate), ", ", str(no_iter), ", ", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction: {score}")

    pass


if __name__ == "__main__":
    PCAIntegrity()
    pass