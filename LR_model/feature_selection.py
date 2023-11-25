from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail, splitData2, reshape_y
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import sys
import math
import copy

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

def feature_selection_filter_corr(X, y, threshold_percent = 0.2, remove_negative = False, minimum = 10, maximum = 0):
    # y should be after reshape
    X_feature = X.T
    no_feature = X_feature.shape[0] #no_feature

    corr_list = []
    corr_dict = {}
    var_y = np.var(y)

    for i in range(no_feature):
        cov = (np.cov(X_feature[i],y)[0][1]) # starting at feature at index 0, column left --> right

        var_Xy = (np.var(X_feature[i])*var_y)
        #print(var_Xy)

        corr = cov/np.sqrt(var_Xy)
        #print(corr)

        corr_list.append(corr)
        corr_dict[corr] = int(i)
    
    corr_list.sort(reverse = True)
    selection = int(no_feature * threshold_percent)

    # minimum = 10, maximum = -1
    if minimum <= no_feature:
        selection = max(minimum, selection)
    if maximum > 0: # meaning there is a maximum, -1 = no max
        selection = min(maximum, selection)

    index_selected = []
    for i in range(selection):
        if remove_negative and corr_list[i] < 0:
            break
        index_selected.append(corr_dict[corr_list[i]])
    index_selected.sort()
    
    return index_selected
    pass

def entropy(data):
    """Compute entropy of data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]
            data structure: [(['vhigh'], 2),(['vhigh'], 0),(['vhigh'], 2), (['vhigh'], 2), (['vhigh'], 1), (['vhigh'], 1)]
                x_i = will be all same value
                y_i = 2 --> int

    Returns:
        entropy of data (float)
    """
    ### YOUR CODE HERE

    entrophy = 0
    sum_x = len(data)
    y = {}
    value_x = data[0][0]
    for tuple in data:
        if tuple[0] != value_x:
            raise ValueError(f"Inconsistent value in entropy() function for {tuple[0]} != {value_x}")
        if tuple[1] in y:
            y[tuple[1]] += 1
        else:
            y[tuple[1]] = 1

    entrophy_equation = ''
    for key in y:
        entrophy += -((y[key]/sum_x) * math.log((y[key]/sum_x),2))
        entrophy_equation = entrophy_equation + f'-({y[key]}/{sum_x} * log({y[key]}/{sum_x}))'
    return round(entrophy, 4)

    ### END YOUR CODE


def gain(data, feature):
    """Compute the gain of data of splitting by feature.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]
        feature: index of feature to split the data (index of x_i list)
            data structure: [(['vhigh', 'high', '3', '4', 'med', 'low'], 2), (['low', 'vhigh', '3', '2', 'small', 'low'], 2)]
                x_i = ['vhigh', 'high', '3', '4', 'med', 'low'] --> list of string
                y_i = 2 --> int

    Returns:
        gain of splitting data by feature
    """
    ### YOUR CODE HERE

    # please call entropy to compute entropy
    # choose feature to compute gain, feature is given as index
    sum_x = len(data)
    new_data = []
    unique_value = set()
    feature_entropy = []
    for record in data:
        new_data.append(([record[0][feature]], record[1])) 
        # get number of unique value
        unique_value.add(record[0][feature])
        # find total entrophy for this feature
        feature_entropy.append((['feature'], record[1]))

    # calculate the gain
    feature_entropy_sum = 0
    for value in unique_value:
        value_entropy = []
        sum_value = 0
        for record in new_data:
            if value == record[0][0]:
                value_entropy.append(record)
                sum_value += 1

        entropy_calculation = entropy(value_entropy)
        feature_entropy_sum += ((sum_value/sum_x) * entropy_calculation)
    
    # return entropy(feature_entropy) - feature_entropy_sum
    return 1 - feature_entropy_sum

    ### END YOUR CODE

def remove_feature(data, feature):
    '''remove a feature from data
    
    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    return:
        data: A list of data points new[(x_0, y_0), ..., (x_n, y_n)]
            where len(new x_i) = len(old x_i) - 1
    '''
    new_data = []
    for record in data:
        x = copy.deepcopy(record[0])
        new_x = np.delete(x, feature)
        new_data.append((new_x, copy.deepcopy(record[1])))
    return new_data

def feature_selection_IG(X, y, threshold_percent = 0.2, minimum = 10, maximum = 0):
    # y should be after reshape
    data_structure = []
    no_data = X.shape[0]
    no_feature = X.shape[1]

    for i in range(no_data):
        data_structure.append((X[i],y[i]))
    
    selection = int(no_feature * threshold_percent)

    # minimum = 10, maximum = -1
    if minimum <= no_feature:
        selection = max(minimum, selection)
    if maximum > 0: # meaning there is a maximum, -1 = no max
        selection = min(maximum, selection)
    
    gain_list = []
    gain_dict = {}

    for i in range(len(data_structure[0][0])):
        gain_feature = gain(data_structure, i)
        gain_list.append((gain_feature, i))
        gain_dict[(gain_feature, i)] = i

    gain_list.sort(reverse=True)

    selection = int(no_feature * threshold_percent)

    index_selected = []
    limit = 0
    for i in gain_list:
        if limit >= selection:
            break
        index_selected.append(gain_dict[i])
        limit += 1
    index_selected.sort()

    return index_selected
    pass

def select_feature(X, index_list):
    X_feature = X.T
    index_list.sort()
    #print(index_selected)

    feature_selected = []
    for i in index_list:
        feature_selected.append(X_feature[i])
    feature_selected = np.array(feature_selected).T
    
    return feature_selected

def testing():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for handwriting dataset ---")
    hand_writing_csv = f"{project_path}/data/handwriting_alzheimers.csv"

    X, y = parseData(hand_writing_csv)
    X = X[:, 1:]
    y = np.where(y == "P", 1, y)
    y = np.where(y == "H", -1, y)
    X = normalize(X) # <-- this is needed before running feature selection , to be fixed
    y = reshape_y(y) # <-- this is needed before running feature selection

    X, y = shuffle(X, y)
    X, y = shuffle(X, y)
    X, y = shuffle(X, y)

    FS_ig = feature_selection_IG(X, y)
    FS_filter = feature_selection_filter_corr(X, y)
    X_filter = select_feature(X, FS_filter)

    X_ig = select_feature(X, FS_ig)
    

    train_x, train_y, _, _, test_x, test_y = splitData(X, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print("\n--- No feature selection ---")
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
    
    train_x, train_y, _, _, test_x, test_y = splitData(X_ig, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print("\n--- IG feature selection ---")
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
    
    train_x, train_y, _, _, test_x, test_y = splitData(X_filter, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print("\n--- filter feature selection ---")
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
    testing()
    pass