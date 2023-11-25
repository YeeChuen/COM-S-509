from feature_reduction import PCA
from helper import parseData, splitData, normalize, score, parseDataBreastCancer, parseDataSpamEmail, reshape_y
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cm

# testing with sklearn library
from sklearn.decomposition import PCA as skPCA

def plotdata3D(X, label, save_name):
    # axes instance
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    fig.add_axes(ax)

    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    x =  X[:,0]
    y =  X[:,1]
    z =  X[:,2]
    # plot

    sc = ax.scatter(x, y, z, s=40, c=label, marker='o', alpha=1, label=label )
    ax.set_xlabel('Feature_1')
    ax.set_ylabel('Feature_2')
    ax.set_zlabel('Feature_3')
    ax.set_title(save_name)


    '''legend1 = ax.legend(*[sc.legend_elements()[0] ,y], 
                        title="Legend", loc='upper right')
    ax.add_artist(legend1)'''

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    # plt.show()

    # save
    plt.savefig(save_name, bbox_inches='tight')

def plotdata2D(X, y, save_name, min_x = 0.1, max_x = 0.1):

    X_df = pd.DataFrame({'Feature_1': X[:,0], 'Feature_2': X[:,1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='label', data=X_df, ax=ax,s=30).set(title=save_name)
    lim = (X.min()-min_x, X.max()+max_x)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(save_name)

def data_visualization3D():
    n_components = 3
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
    y = np.where(y == "P", 1, y)
    y = np.where(y == "H", -1, y)
    y = reshape_y(y)
    data = PCA(X, k=n_components)
    #print(data)
    plotdata3D(data, y, '3D Alzheimers data visualization PCA 1')
   
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    #print(tsne_result)
    plotdata3D(tsne_result, y, '3D Alzheimers data visualization TSNE')
    
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata3D(X_pca_sklearn, y, '3D Alzheimers data visualization PCA 2')

    print("")
    print("--- data visualization for breast cancer dataset ---")
    breast_cancer_csv = f"{project_path}/data/breast-cancer.csv"
    X, y = parseDataBreastCancer(breast_cancer_csv)
    X = normalize(X)
    y = np.where(y == "M", 1, y)
    y = np.where(y == "B", -1, y)
    y = reshape_y(y)
    data = PCA(X, k=n_components)
    #print(data)
    plotdata3D(data, y, '3D Breast cancer data visualization PCA 1')

    
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata3D(tsne_result, y, '3D Breast cancer data visualization TSNE')
    
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata3D(X_pca_sklearn, y, '3D Breast cancer data visualization PCA 2')

    print("")
    print("--- data visualization for spam email dataset ---")
    spam_email_csv = f"{project_path}/data/spam_email_dataset.csv"
    X , y = parseDataSpamEmail(spam_email_csv)
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X, k=n_components)
    plotdata3D(data, y, '3D Spam email data visualization PCA 1')
    
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata3D(tsne_result, y, '3D Spam email data visualization TSNE')

    # test sklearn PCA
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata3D(X_pca_sklearn, y, '3D Spam email data visualization PCA 2')

    print("")
    print("--- data visualization for water potability dataset ---")
    water_potability_csv = f"{project_path}/data/water_potability.csv"
    X, y = parseData(water_potability_csv)
    X = normalize(X, nan = 'median')
    y = reshape_y(y)
    data = PCA(X, k=n_components)
    plotdata3D(data, y, '3D Water potability data visualization PCA 1')
    
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata3D(tsne_result, y, '3D Water potability data visualization TSNE')
    
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata3D(X_pca_sklearn, y, '3D Water potability data visualization PCA 2')

def data_visualization2D():
    n_components = 2
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
    data = PCA(X, k = n_components)
    plotdata2D(data, y, 'Alzheimers data visualization PCA 1', min_x = 0.05, max_x = 0.05)
   
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Alzheimers data visualization TSNE', min_x = 0.05, max_x = 0.05)
    
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata2D(X_pca_sklearn, y, 'Alzheimers data visualization PCA 2', min_x = 0.05, max_x = 0.05)

    print("")
    print("--- data visualization for breast cancer dataset ---")
    breast_cancer_csv = f"{project_path}/data/breast-cancer.csv"
    X, y = parseDataBreastCancer(breast_cancer_csv)
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X)
    plotdata2D(data, y, 'Breast cancer data visualization PCA 1', min_x = 0.05, max_x = 0.05)

    
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Breast cancer data visualization TSNE', min_x = 0.05, max_x = 0.05)
    
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata2D(X_pca_sklearn, y, 'Breast cancer data visualization PCA 2', min_x = 0.05, max_x = 0.05)

    print("")
    print("--- data visualization for spam email dataset ---")
    spam_email_csv = f"{project_path}/data/spam_email_dataset.csv"
    X , y = parseDataSpamEmail(spam_email_csv)
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X)
    plotdata2D(data, y, 'Spam email data visualization PCA 1', min_x = 0.05, max_x = 0.05)
    
    
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Spam email data visualization TSNE', min_x = 0.05, max_x = 0.05)

    # test sklearn PCA
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata2D(X_pca_sklearn, y, 'Spam email data visualization PCA 2', min_x = 0.05, max_x = 0.05)

    print("")
    print("--- data visualization for water potability dataset ---")
    water_potability_csv = f"{project_path}/data/water_potability.csv"
    X, y = parseData(water_potability_csv)
    X = normalize(X)
    y = reshape_y(y)
    data = PCA(X)
    plotdata2D(data, y, 'Water potability data visualization PCA 1', min_x = 0.05, max_x = 0.05)
    
    
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    plotdata2D(tsne_result, y, 'Water potability data visualization TSNE', min_x = 0.05, max_x = 0.05)
    
    pca = skPCA(n_components = n_components).fit(X)
    X_pca_sklearn = pca.transform(X) # Apply dimensionality reduction to X.
    #plotdata2D(X_pca_sklearn, y, 'Water potability data visualization PCA 2', min_x = 0.05, max_x = 0.05)

    print("")
    pass

if __name__ == "__main__":
    data_visualization2D()
    data_visualization3D()
    pass