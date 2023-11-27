from logistic_regression import LogisticRegression
from helper import parseData, splitData, normalize, ensemble_wrapper, poly_kernel, rbf_kernel, sigmoid_kernel, bagging,boosting, get_datasets, data_reconstruct, applyCorruption
from sklearn.utils import shuffle
import numpy as np
from feature_reduction import PCA
from feature_selection import feature_selection_filter_corr, select_feature
from hyperparam_tuning import hyperparam_tuning


def write(s):
    save = 'D:/Files/ISU work/Computer Science Program/2023/Fall 2023/COM S 573/Term Project/COM-S-573/LR_model/results.txt'
    with open(save, 'a') as f:
        f.write(f'{s}\n')

# training and testing model functions
def training_test_model_bagging(
        train_x, train_y, val_x, val_y, test_x, test_y, 
        l_rate1 = 0.001, no_iter1 = 100, best_prob1 = 0.5, l_rate2 = 0.001, no_iter2 = 100, best_prob2 = 0.5, 
        batch_size = 10, C = 1, num_models = 10):

        ws, bs, ls = bagging(train_x, train_y, val_x, val_y, C, batch_size, l_rate1, no_iter1, num_models, int(train_x.shape[0] / 3), 
                            model = LogisticRegression, model_type = 'binomial')
        score = ensemble_wrapper(test_x, test_y, ws, bs, False, 0.5,
                               model = LogisticRegression, model_type = 'binomial')
        write(f"scratch binom LRmodel prediction: {score}")
        
        ws, bs, ls = bagging(train_x, train_y, val_x, val_y, C, batch_size, l_rate2, no_iter2, num_models, int(train_x.shape[0] / 3), 
                            model = LogisticRegression, model_type = 'multinomial')
        score = ensemble_wrapper(test_x, test_y, ws, bs, False, 0.5,
                               model = LogisticRegression, model_type = 'multinomial')
        write(f"scratch multi LRmodel prediction: {score}")

def training_test_model_boosting(
        train_x, train_y, val_x, val_y, test_x, test_y, 
        l_rate1 = 0.001, no_iter1 = 100, best_prob1 = 0.5, l_rate2 = 0.001, no_iter2 = 100, best_prob2 = 0.5, 
        batch_size = 10, num_models = 10):

        ws, bs, ls, lWs = boosting(train_x, train_y, val_x, val_y, batch_size, l_rate1, no_iter1, num_models, int(train_x.shape[0] / 3), 
                            model = LogisticRegression, model_type = 'binomial')
        score = ensemble_wrapper(test_x, test_y, ws, bs, True, 0.5, accs = lWs,
                               model = LogisticRegression, model_type = 'binomial')
        write(f"scratch binom LRmodel prediction: {score}")
        
        ws, bs, ls, lWs = boosting(train_x, train_y, val_x, val_y, batch_size, l_rate2, no_iter2, num_models, int(train_x.shape[0] / 3), 
                            model = LogisticRegression, model_type = 'multinomial')
        score = ensemble_wrapper(test_x, test_y, ws, bs, True, 0.5, accs = lWs,
                               model = LogisticRegression, model_type = 'multinomial')
        write(f"scratch multi LRmodel prediction: {score}")

def training_test_model(
        train_x, train_y, val_x, val_y, test_x, test_y, 
        l_rate1 = 0.001, no_iter1 = 100, best_prob1 = 0.5, l_rate2 = 0.001, no_iter2 = 100, best_prob2 = 0.5):
    model = LogisticRegression(l_rate2, no_iter2)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob= best_prob1)
    write(f"scratch binom LRmodel prediction: {score}")

    model = LogisticRegression(l_rate1, no_iter1, classifier="multinomial")
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y, prob= best_prob2)
    write(f"scratch multi LRmodel prediction: {score}")
'''
Normalization
Kernelization
Feature selection
Feature reduction
Shuffling
Bagging
Boosting
Hyperparameter-tuning
Data-reconstruction
Combination of the above techniques
'''
def basecase():
    write("\n --- base case ---")
    #[[X_hw, y_hw], [X_bc, y_bc], [X_se, y_se], [X_wp, y_wp]]
    # index:    0-alzheimers, 1-breastcancer, 2-spamemail, 3-water-potability
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]
        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def normalization():
    write("\n --- normalization ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        X = normalize(X)

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def poly_kernelization():
    write("\n --- poly kernelization ---")
    #[[X_hw, y_hw], [X_bc, y_bc], [X_se, y_se], [X_wp, y_wp]]
    # index:    0-alzheimers, 1-breastcancer, 2-spamemail, 3-water-potability
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        X = poly_kernel(X, 1, 1, 2)

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def rbf_kernelization():
    write("\n --- rbf kernelization ---")
    #[[X_hw, y_hw], [X_bc, y_bc], [X_se, y_se], [X_wp, y_wp]]
    # index:    0-alzheimers, 1-breastcancer, 2-spamemail, 3-water-potability
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        X = rbf_kernel(X, 1)

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def sigmoid_kernelization():
    write("\n --- sigmoid kernelization ---")
    #[[X_hw, y_hw], [X_bc, y_bc], [X_se, y_se], [X_wp, y_wp]]
    # index:    0-alzheimers, 1-breastcancer, 2-spamemail, 3-water-potability
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        X = sigmoid_kernel(X, 1, 1)

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def feature_selection():
    write("\n --- feature selection ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        try:
            # X = normalize(X) # <-- odd issue where this needs to be run first
            FS_filter = feature_selection_filter_corr(X, y)
            X_filter = select_feature(X, FS_filter)

            train_x, train_y, val_x, val_y, test_x, test_y = splitData(X_filter, y, 0.8, 0.0, 0.2)
            training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
        except:
            X = normalize(X) # <-- odd issue where this needs to be run first
            FS_filter = feature_selection_filter_corr(X, y)
            X_filter = select_feature(X, FS_filter)

            train_x, train_y, val_x, val_y, test_x, test_y = splitData(X_filter, y, 0.8, 0.0, 0.2)
            training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def feature_reduction():
    write("\n --- feature reduction ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        # X = normalize(X) # <-- odd issue where this needs to be run first
        try:
            X_pca = PCA(X, k=X.shape[1] * 0.2)

            train_x, train_y, val_x, val_y, test_x, test_y = splitData(X_pca, y, 0.8, 0.0, 0.2)
            training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
        except:
            X = normalize(X) # <-- odd issue where this needs to be run first
            X_pca = PCA(X, k=X.shape[1] * 0.2)

            train_x, train_y, val_x, val_y, test_x, test_y = splitData(X_pca, y, 0.8, 0.0, 0.2)
            training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def shuffling():
    write("\n --- shuffling ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]
        
        X, y = shuffle(X, y)
        X, y = shuffle(X, y)
        X, y = shuffle(X, y)

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def bagging_model():
    write("\n --- bagging ---")
    datasets = get_datasets()

    '''
    key = 'Spam Email'
    dataset = datasets[key]
    write(f"\nModel performance for: {key}")
    X = dataset[0]
    y = dataset[1]

    train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)

    training_test_model_bagging(train_x, train_y, val_x, val_y, test_x, test_y)
    '''

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)

        training_test_model_bagging(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def boosting_model():
    write("\n --- boosting ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)

        training_test_model_boosting(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def hyperparameter_tuning():
    write("\n --- hyperparameter tuning ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        
        l_rate1, no_iter1, best_prob1 = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
        l_rate2, no_iter2, best_prob2 = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)

        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y, 
                            l_rate1, no_iter1, best_prob1, l_rate2, no_iter2, best_prob2)
    pass

def data_reconstruction():
    write("\n --- data reconstruction ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        
        X = dataset[0]
        y = dataset[1]

        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)

        #data corruption simulation for robustness
        train_x = applyCorruption(train_x, 0.5, 0.5, True)
        test_x = applyCorruption(test_x, 0.5, 0.5, True)

        #data reconstruction for corrupted data
        train_x = data_reconstruct(train_x, 0.25)
        test_x = data_reconstruct(test_x, 0.25)
    
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

def combination():
    write("\n --- combination ---")
    datasets = get_datasets()

    for key in datasets:
        dataset = datasets[key]
        write(f"\nModel performance for: {key}")
        X = dataset[0]
        y = dataset[1]

        X, y = shuffle(X, y)
        X, y = shuffle(X, y)
        X, y = shuffle(X, y)
        write("--- basecase ---")
        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)

        X = normalize(X)
        write("--- add normalization ---")
        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)

        FS_filter = feature_selection_filter_corr(X, y, remove_negative=True, minimum = X.shape[1])
        X_filter = select_feature(X, FS_filter)
        write("--- add feature selection ---")
        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X_filter, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
        
        X_pca = PCA(X_filter, k=2)
        write("--- add feature reduction w/ FS ---")
        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X_pca, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
        
        write("--- add hyperparameter tuning ---")
        l_rate1, no_iter1, best_prob1 = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
        l_rate2, no_iter2, best_prob2 = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)

        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y, 
                            l_rate1, no_iter1, best_prob1, l_rate2, no_iter2, best_prob2)

        write("--- add bagging ---")
        training_test_model_bagging(train_x, train_y, val_x, val_y, test_x, test_y)
        
        write("--- add boosting ---")
        training_test_model_boosting(train_x, train_y, val_x, val_y, test_x, test_y)


        X_pca = PCA(X, k=X.shape[1] * 0.2)
        write("--- add feature reduction w/o FS ---")
        train_x, train_y, val_x, val_y, test_x, test_y = splitData(X_pca, y, 0.8, 0.0, 0.2)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y)
        
        write("--- add hyperparameter tuning w/o FS ---")
        l_rate1, no_iter1, best_prob1 = hyperparam_tuning(LogisticRegression, "multinomial", train_x, train_y)
        l_rate2, no_iter2, best_prob2 = hyperparam_tuning(LogisticRegression, "binomial", train_x, train_y)
        training_test_model(train_x, train_y, val_x, val_y, test_x, test_y, 
                            l_rate1, no_iter1, best_prob1, l_rate2, no_iter2, best_prob2)

        write("--- add bagging w/o FS ---")
        training_test_model_bagging(train_x, train_y, val_x, val_y, test_x, test_y)
        
        write("--- add boosting w/o FS ---")
        training_test_model_boosting(train_x, train_y, val_x, val_y, test_x, test_y)
    pass

if __name__ == "__main__":
    # basecase handwriting get 100% accuracy because
    # the pred value is too low, it turns into Nan
    # which predicts -1, and since there is no shuffle, all -1 data is at the end, hence 100%
    # this issue occurs similarly, until we test with it with shuffle. 

    # shuffle is needed to ensure realistic testing, imo
    basecase()              # 1 CHECK

    normalization()          # 2 CHECK
    poly_kernelization()     # 3 <-- ignore for now
    rbf_kernelization()     # 3 <-- ignore for now
    sigmoid_kernelization()     # 3 <-- ignore for now

    # current fix for below 2, is to try without normalize, except with normalize, 
    feature_selection()     # 4 <-- odd issue where it needs normalization first, 
    feature_reduction()     # 5 <-- similarly with #3

    shuffling()             # 6 CHECK

    # model will go too low of float, turns into Nan, and predicts all -1
    bagging_model()          # 7 CHECK
    boosting_model()         # 8 CHECK

    hyperparameter_tuning() # 9 CHECK
    data_reconstruction()   # 10 CHECK

    for i in range(10):
        write("Round ", i)
        combination()           # 11 No combinations yet.
    pass