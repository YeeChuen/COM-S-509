import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import CustomLogisticRegression
from logistic_regression import LogisticRegression as MyLogisticRegression, ShortLogisticRegression

def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y

x, y = sklearn_to_df(load_breast_cancer())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

lr = CustomLogisticRegression()
lr.fit(x_train, y_train, epochs=150)
score = lr.score(x_test, y_test)
print(score)

model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
print(score)

model = MyLogisticRegression(0.1, 1000)
model.fit(x_train.values, y_train.values)
score = model.score(x_test.values, y_test.values)
print(f"scratch model prediction: {score}")

model = ShortLogisticRegression(0.1, 1000)
model.fit(x_train.values, y_train.values)
score = model.score(x_test.values, y_test.values)
print(f"short model prediction: {score}")