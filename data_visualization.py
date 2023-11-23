from feature_reduction import PCA
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail, reshape_y
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def plotdata2D(X, y, save_name, min_x = 0.1, max_x = 0.1):

    X_df = pd.DataFrame({'pca_1': X[:,0], 'pca_2': X[:,1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='pca_1', y='pca_2', hue='label', data=X_df, ax=ax,s=30)
    lim = (X.min()-min_x, X.max()+max_x)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(save_name)

def data_visualization():
    path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573/data_visualization'
    project_path = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573'
    print(os.getcwd())
    os.chdir(path)

    print("")
    print("--- data visualization for handwriting dataset ---")
    hand_writing_csv = f"{project_path}/data/handwriting_alzheimers.csv"
    X, y = parseData(hand_writing_csv)
    X = X[:, 1:]
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X)
    plotdata2D(data, y, 'Alzheimers data visualization PCA', min_x = 0.05, max_x = 0.05)
   
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Alzheimers data visualization TSNE', min_x = 0.05, max_x = 0.05)

    print("")
    print("--- data visualization for breast cancer dataset ---")
    breast_cancer_csv = f"{project_path}/data/breast-cancer.csv"
    X, y = parseDataBreastCancer(breast_cancer_csv)
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X)
    plotdata2D(data, y, 'Breast cancer data visualization PCA', min_x = 0.05, max_x = 0.05)
    
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Breast cancer data visualization TSNE', min_x = 0.05, max_x = 0.05)

    print("")
    print("--- data visualization for spam email dataset ---")
    spam_email_csv = f"{project_path}/data/spam_email_dataset.csv"
    X , y = parseDataSpamEmail(spam_email_csv)
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X)
    plotdata2D(data, y, 'Spam email data visualization PCA', min_x = 0.05, max_x = 0.05)
    
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Spam email data visualization TSNE', min_x = 0.05, max_x = 0.05)

    print("")
    print("--- data visualization for water potability dataset ---")
    water_potability_csv = f"{project_path}/data/water_potability.csv"
    X, y = parseData(water_potability_csv)
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X)
    plotdata2D(data, y, 'Water potability data visualization PCA', min_x = 0.05, max_x = 0.05)
    
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Water potability data visualization TSNE', min_x = 0.05, max_x = 0.05)

    print("")
    pass

if __name__ == "__main__":
    data_visualization()
    pass