from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, score
from sklearn.utils import shuffle
import numpy as np

# test logistic regresison model
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from logistic_regression import CustomLogisticRegression, ShortLogisticRegression

def trySKlearn():
    print("")
    print("--- comparing model accuracy for digit dataset ---")
    
    digits = load_digits()  
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
    print(x_train.shape)
    print(y_train.shape)

    model = LogisticRegression(0.001, 150)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_train, y_train)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)
    score = lr.score(x_train, y_train)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    print(f"github 2 model prediction: {score}")

    print("")

def handwriting_dataset():
    print("")
    print("--- comparing model accuracy for handwriting dataset ---")
    hand_writing_csv = "./data/handwriting_alzheimers.csv"

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


    model = LogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(train_x, train_y)
    print(f"scratch model prediction: {score}")

    logisticRegr = skLogisticRegression()
    logisticRegr.fit(train_x, train_y)
    score = logisticRegr.score(train_x, train_y)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(train_x, train_y, epochs=150)
    score = lr.score(train_x, train_y)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 150)
    model.fit(train_x, train_y)
    score = model.score(train_x, train_y)
    print(f"github 2 model prediction: {score}")

    print("")
    pass

def breastcancer_dataset():
    breast_cancer_csv = "./data/breast-cancer.csv"
    pass

def spamemail_dataset():
    spam_email_csv = "./data/spam_email_dataset.csv"
    pass

def waterpotability_dataset():
    water_potability_csv = "./data/water_potability.csv"
    pass

if __name__ == "__main__":
    handwriting_dataset()
    trySKlearn()
    pass