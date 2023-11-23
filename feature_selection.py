from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail, splitData2
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

def feature_selection_filter():


    pass

def testing():
    print("")
    print("--- comparing model accuracy for handwriting dataset ---")
    hand_writing_csv = "./data/handwriting_alzheimers.csv"

    X, y = parseData(hand_writing_csv)


    df = pd.DataFrame(X)
    print(df)


    pass


if __name__ == "__main__":
    testing()
    pass