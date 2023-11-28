from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail, splitData2
from sklearn.utils import shuffle
import numpy as np

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
import copy

def hyperparam_tuning(LRmodel, model_classifier, X, y, n_fold = 5, percent = False):
    # print("Hyperparameter tuning ... ")

    # train_x, train_y, _, _, test_x, test_y = splitData2(X, y, 0.8, 0, 0.2)
    '''print(X.shape)
    print(y.shape)'''

    x_fold = []
    y_fold = []
    length = X.shape[0]
    fold = int(length/5)
    for i in range(n_fold - 1):
        x_fold.append(X[fold*i:fold*(i+1) ,:])
        y_fold.append(y[fold*i:fold*(i+1)])
    x_fold.append(X[fold*(n_fold - 1):length ,:])
    y_fold.append(y[fold*(n_fold - 1):length])

    best = 0
    best_lr = 1
    best_iter = 100
    best_prob = 0.5
    for i in range(10):
        rate = 10
        iter = 100
        for _ in range(i):
            rate *= 0.1
        for j in range(10):
            if percent:
                for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

                    score = 0
                    for k in range(len(x_fold)):  
                        model = LRmodel(rate, iter, classifier=model_classifier)
                        deep_X = copy.deepcopy(X)
                        deep_y = copy.deepcopy(y)
                        deep_X_list = deep_X.tolist()
                        deep_y_list = deep_y.tolist()

                        remove_X = x_fold[k]

                        for m in range(len(deep_X_list)):
                            if deep_X_list[m] in remove_X:
                                print("removed: ", m)
                                deep_X_list.pop(m)
                                deep_y_list.pop(m)
                        
                        model.fit(np.array(deep_X_list), np.array(deep_y_list))

                        score += model.score(x_fold[k], y_fold[k], prob = threshold)
                    score /= len(x_fold)

                    # print(f"Score {score} for lr: {rate}, iter: {iter}, prob: {threshold}")

                    if score > best:
                        best = score
                        best_lr = rate
                        best_iter = iter
                        best_prob = threshold
            else:
                score = 0
                for k in range(len(x_fold)):
                    fold_test_X = x_fold[k]
                    fold_test_y = y_fold[k]
                    if k == 0:
                        start = 1
                        fold_train_X = x_fold[start]
                        fold_train_y = y_fold[start]

                    else:
                        start = 0
                        fold_train_X = x_fold[start]
                        fold_train_y = y_fold[start]

                    for m in range(start + 1, len(x_fold)):
                        if m == k:
                            continue
                        else:
                            fold_train_X = np.concatenate((fold_train_X, x_fold[m]), axis=0)
                            fold_train_y = np.concatenate((fold_train_y, y_fold[m]), axis=0)
                        
                        '''print("add ", m)
                        print(fold_train_X.shape)
                        print(fold_train_y.shape)'''

                    model = LRmodel(rate, iter, classifier=model_classifier)                        
                    model.fit(fold_train_X, fold_train_y)

                    score += model.score(fold_test_X, fold_test_y)
                score /= len(x_fold)

                # print(f"Score {score} for lr: {rate}, iter: {iter}, prob: {threshold}")

                if score > best:
                    best = score
                    best_lr = rate
                    best_iter = iter

            iter += 100
    
    #print("learning rate:",str(best_lr), ", No iterations:", str(best_iter), ", Probability threshold:", str(best_prob))
    return best_lr, best_iter, best_prob

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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    
    print(x_train.values.shape)
    print(y_train.values.shape)
    

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "multinomial", x_train.values, y_train.values)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values, prob = best_prob)
    print(f"scratch multi model prediction: {score}")
    
    
    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", x_train.values, y_train.values)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values, prob = best_prob)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_test,y_test)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)
    score = lr.score(x_test,y_test)
    print(f"github 1 model prediction: {score}")
    
    print("")
    pass

def trySKlearn():
    print("")
    print("--- comparing model accuracy for sklearn digit dataset ---")
    
    digits = load_digits()  
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
    print(x_train.shape)
    print(y_train.shape)
    
    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "multinomial", x_train, y_train)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test, prob = best_prob)
    print(f"scratch multi model prediction: {score}")

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", x_train, y_train)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test, prob = best_prob)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_test, y_test)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)
    score = lr.score(x_test, y_test)
    print(f"github 1 model prediction: {score}")

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
    X = normalize(X)

    #Due to the data have a low sample count, depending on the distribution of the shuffle
    #accuracy can be extremely poor
    #While shuffling help negate this, we need another technique to increase sample count
    X, y = shuffle(X, y)
    X, y = shuffle(X, y)
    X, y = shuffle(X, y)

    train_x, train_y, _, _, test_x, test_y = splitData(X, y, 0.8, 0, 0.2)
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
    
    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch multi model prediction: {score}")

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction: {score}")

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

    train_x, train_y, _, _, test_x, test_y = splitData(X, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print(train_y.shape)
    
    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch multi model prediction: {score}")

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")

    print("")
    pass

def spamemail_dataset():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for spam email dataset ---")
    spam_email_csv = f"{project_path}/data/spam_email_dataset.csv"
    X , y = parseDataSpamEmail(spam_email_csv)
    X = normalize(X)

    train_x, train_y, _, _, test_x, test_y = splitData(X, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print(train_y.shape)

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch multi model prediction: {score}")

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")
    
    print("")
    pass

def waterpotability_dataset():
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print("")
    print("--- comparing model accuracy for water potability dataset ---")
    water_potability_csv = f"{project_path}/data/water_potability.csv"
    
    X, y = parseData(water_potability_csv)
    X = normalize(X)

    train_x, train_y, _, _, test_x, test_y = splitData(X, y, 0.8, 0, 0.2)
    print(train_x.shape)
    print(train_y.shape)

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch multi model prediction: {score}")

    l_rate, no_iter, best_prob = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)
    print("learning rate:",str(l_rate), ", No iterations:", str(no_iter), ", Probability threshold:", str(best_prob))
    model = LogisticRegression(l_rate, no_iter)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob = best_prob)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")

    print("")
    pass

if __name__ == "__main__":
    print("\nhyper parameter tuning")

    #trySKlearn()
    #sklearnDataset()
    handwriting_dataset()
    breastcancer_dataset()
    waterpotability_dataset()
    spamemail_dataset()

    pass