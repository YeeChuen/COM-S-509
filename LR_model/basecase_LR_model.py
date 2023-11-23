from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail
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
from LRmodel import sklearnDataset
import pandas as pd

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
    
    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"scratch multi model prediction: {score}")
    
    model = LogisticRegression(0.1, 150)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_test,y_test)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)
    score = lr.score(x_test,y_test)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"github 2 model prediction: {score}")
    
    model = PELogisticRegression(0.1, 150)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"PE website model prediction: {score}")
    
    print("")
    pass

def trySKlearn():
    print("")
    print("--- comparing model accuracy for sklearn digit dataset ---")
    
    digits = load_digits()  
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
    print(x_train.shape)
    print(y_train.shape)
    
    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"scratch multi model prediction: {score}")

    model = LogisticRegression(0.1, 150)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_test, y_test)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)
    score = lr.score(x_test, y_test)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"github 2 model prediction: {score}")

    model = PELogisticRegression(0.1, 150)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"PE website model prediction: {score}")

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
    
    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch multi model prediction: {score}")

    model = LogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"github 2 model prediction: {score}")
    
    model = PELogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"PE website model prediction: {score}")

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
    
    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch multi model prediction: {score}")

    model = LogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"github 2 model prediction: {score}")
    
    model = PELogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"PE website model prediction: {score}")

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

    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch multi model prediction: {score}")

    model = LogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"github 2 model prediction: {score}")
    
    model = PELogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"PE website model prediction: {score}")
    
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

    model = LogisticRegression(0.1, 150, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch multi model prediction: {score}")

    model = LogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(test_x, test_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(test_x, test_y)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"github 2 model prediction: {score}")
    
    model = PELogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"PE website model prediction: {score}")

    print("")
    pass

if __name__ == "__main__":
    print("\nBasecase, hyper parameter are set to:\nlearning rate: 0.1\niterations: 150")

    trySKlearn()
    sklearnDataset()
    handwriting_dataset()
    breastcancer_dataset()
    waterpotability_dataset()
    spamemail_dataset()

    pass