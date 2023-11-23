import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import CustomLogisticRegression
from logistic_regression import LogisticRegression as MyLogisticRegression, ShortLogisticRegression, PELogisticRegression

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
    
    model = MyLogisticRegression(0.1, 1000)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"scratch model prediction: {score}")

    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_test,y_test)
    print(f"sklearn prediction: {score}")

    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)
    score = lr.score(x_test,y_test)
    print(f"github 1 model prediction: {score}")
    
    model = ShortLogisticRegression(0.1, 1000)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"github 2 model prediction: {score}")
    
    model = PELogisticRegression(0.1, 1000)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"PE website model prediction: {score}")
    
    print("")
    pass

if __name__ == "__main__":
    print("")
    x, y = sklearn_to_df(load_breast_cancer())

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    
    model = MyLogisticRegression(0.01, 1000, classifier="multinomial")
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"scratch multi model prediction: {score}")

    
    model = MyLogisticRegression(0.01, 500)
    model.fit(x_train.values, y_train.values)
    score = model.score(x_test.values, y_test.values)
    print(f"scratch binomial model prediction: {score}")