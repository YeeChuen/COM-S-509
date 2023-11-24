from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail, splitData2, reshape_y
from sklearn.utils import shuffle
import numpy as np
from feature_reduction import PCA
from feature_selection import feature_selection_filter_corr, select_feature
from hyperparam_tuning import hyperparam_tuning

# encode string to number
from sklearn.preprocessing import OneHotEncoder

# test logistic regresison model
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from logistic_regression import CustomLogisticRegression, ShortLogisticRegression, PELogisticRegression
import pandas as pd
from tqdm import tqdm

def training_all_model(X, y, l_rate1, no_iter1, best_prob1, l_rate, no_iter, best_prob):
    train_x, train_y, _, _, test_x, test_y = splitData2(X, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print(train_y.shape)

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")
    
    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch multi model prediction: {score}")
    
    model = LogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch model prediction: {score}")
    
    l_rate1, no_iter1, best_prob1 = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
    model = LogisticRegression(l_rate1, no_iter1, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob1)
    print(f"scratch multi model prediction(w/HT): {score}")

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction(w/HT): {score}")
    
    FS_filter = feature_selection_filter_corr(X, y, threshold_percent=0.5, minimum = 10, maximum=30)
    X_filter = select_feature(X, FS_filter)
    X_pca = PCA(X_filter, k = int(X_filter.shape[1] * 0.5))
    train_x, train_y, _, _, test_x, test_y = splitData2(X_pca, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print(train_y.shape)
    
    model = LogisticRegression(l_rate1, no_iter1, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob1)
    print(f"scratch multi model prediction(w/HT,FS,FR): {score}")

    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction(w/HT,FS,FR): {score}")

    X_pca = PCA(X, k = 2)
    train_x, train_y, _, _, test_x, test_y = splitData2(X_pca, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print(train_y.shape)
    
    l_rate1, no_iter1, best_prob1 = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
    model = LogisticRegression(l_rate1, no_iter1, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob1)
    print(f"scratch multi model prediction(w/HT,FS,FR): {score}")

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction(w/HT,FS,FR): {score}")

def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y

def sklearnDataset():
    print("")
    print("--- comparing model accuracy for sklearn breast cancer dataset ---")

    x, y = sklearn_to_df(load_breast_cancer())

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    train_x = train_x.values
    test_x = test_x.values
    train_y = train_y.values
    test_y = test_y.values

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")
    
    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch multi model prediction: {score}")
    
    model = LogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch model prediction: {score}")
    
    l_rate1, no_iter1, best_prob1 = 0.001, 1000, 0.5
    print("learning rate:",str(l_rate1), ", No iterations:", str(no_iter1), ", Probability threshold:", str(best_prob1))
    model = LogisticRegression(l_rate1, no_iter1, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob1)
    print(f"scratch multi model prediction(w/HT): {score}")

    l_rate, no_iter, best_prob = 0.0001, 400, 0.3
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction(w/HT): {score}")
    
    X = x.values
    y = y.values
    FS_filter = feature_selection_filter_corr(X, y, threshold_percent=0.5, minimum = 10, maximum=30)
    X_filter = select_feature(X, FS_filter)
    X_pca = PCA(X_filter, k = int(X_filter.shape[1] * 0.5))
    train_x, train_y, _, _, test_x, test_y = splitData2(X_pca, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print(train_y.shape)
    
    model = LogisticRegression(l_rate1, no_iter1, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob1)
    print(f"scratch multi model prediction(w/HT,FS,FR): {score}")

    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction(w/HT,FS,FR): {score}")
    
    print("")
    pass

def trySKlearn():
    print("")
    print("--- comparing model accuracy for sklearn digit dataset ---")
    digits = load_digits()  

    training_all_model(digits.data, digits.target, 0.001, 100, 0.5, 0.01, 100, 0.1)

    print("")

def handwriting_dataset():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for handwriting dataset ---")
    hand_writing_csv = f"{project_path}/data/handwriting_alzheimers.csv"

    X, y = parseData(hand_writing_csv)
    X = X[:, 1:]
    y = np.where(y == "P", 1, y)
    y = np.where(y == "H", -1, y)
    y = reshape_y(y)
    X = normalize(X)

    #Due to the data have a low sample count, depending on the distribution of the shuffle
    #accuracy can be extremely poor
    #While shuffling help negate this, we need another technique to increase sample count
    X, y = shuffle(X, y)
    X, y = shuffle(X, y)
    X, y = shuffle(X, y)

    training_all_model(X, y, 0.1, 100, 0.4, 0.001, 100, 0.1)

    print("")
    pass

def breastcancer_dataset():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for breast cancer dataset ---")
    breast_cancer_csv = f"{project_path}/data/breast-cancer.csv"
    
    X, y = parseDataBreastCancer(breast_cancer_csv)
    X = normalize(X)
    
    y = np.where(y == "M", 1, y)
    y = np.where(y == "B", -1, y)
    y = reshape_y(y)

    training_all_model(X, y, 0.1, 100, 0.5, 0.01, 100, 0.1)

    print("")
    pass


def waterpotability_dataset():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for water potability dataset ---")
    water_potability_csv = f"{project_path}/data/water_potability.csv"
    
    X, y = parseData(water_potability_csv)
    X = X[:,1:]
    Xzero = normalize(X)
    y = reshape_y(y)

    training_all_model(Xzero, y, 0.1, 100, 0.5, 0.1, 100, 0.4)

    Xmean = normalize(X, nan = 'mean')
    print("--> water potability with mean Nan")
    training_all_model(Xmean, y, 0.1, 100, 0.5, 0.1, 100, 0.4)
    
    Xmedian = normalize(X, nan = 'median')
    print("--> water potability with median Nan")
    training_all_model(Xmedian, y, 0.1, 100, 0.5, 0.1, 100, 0.4)

    print("")
    pass

def spamemail_dataset():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for spam email dataset ---")
    spam_email_csv = f"{project_path}/data/spam_email_dataset.csv"
    X , y = parseDataSpamEmail(spam_email_csv)
    X = normalize(X)
    y = reshape_y(y)

    training_all_model(X, y, 0.1, 100, 0.5, 0.1, 100, 0.4)

    print("")
    pass

if __name__ == "__main__":
    print("\nhyper parameter tuning")

    trySKlearn()
    sklearnDataset()
    handwriting_dataset()
    breastcancer_dataset()
    waterpotability_dataset()
    spamemail_dataset()

    pass