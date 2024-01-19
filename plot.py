import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# for commit testing

def readfile(folder):
    with open(folder, 'r') as f:
        to_return = []
        file_content = f.readlines()
        for file in file_content:
            file = file.replace("\n","")
            to_return.append(file.split(","))
        return to_return

def plot_bar_individual(file_content_list, save_name, dataset_name, model_type, combination = None):

    dataset_labels = []
    combination_test = {}
    combination_train = {}

    labels = []
    legend = []
    for content in file_content_list:
        if content[1] not in legend:
            legend.append(content[1])

    techniques_test = {}
    techniques_train = {}
    test = []
    train = []

    for content in file_content_list:
        if content[1] != dataset_name:
            continue

        if content[0] not in labels:
            labels.append(content[0])
        if content[0] not in techniques_test:
            techniques_test[content[0]] = []
        if content[0] not in techniques_train:
            techniques_train[content[0]] = []
        techniques_test[content[0]].append(float(content[2])*100)
        techniques_train[content[0]].append(float(content[3])*100)

    for technique_key in labels:
        average_test = sum(techniques_test[technique_key]) / len(techniques_test[technique_key]) 
        test.append(average_test)

        average_train = sum(techniques_train[technique_key]) / len(techniques_train[technique_key]) 
        train.append(average_train)

    # plot with combination if provided
    if combination:
        for content in combination:
            if content[1] not in dataset_labels:
                dataset_labels.append(content[1])
            if content[1] not in combination_test:
                combination_test[content[1]] = []
            if content[1] not in combination_train:
                combination_train[content[1]] = []
            combination_test[content[1]].append(float(content[2])*100)
            combination_train[content[1]].append(float(content[3])*100)

        mean = {}
        mean['test'] = {}
        mean['train'] = {}

        for dataset in dataset_labels:
        #labels = ['G1', 'G2', 'G3', 'G4', 'G5']
            combination_test[dataset].sort()
            combination_train[dataset].sort()
            average_test = sum(combination_test[dataset]) / len(combination_test[dataset]) 
            mean['test'][dataset] = average_test
            average_train = sum(combination_train[dataset]) / len(combination_train[dataset]) 
            mean['train'][dataset] = average_train
        
        #print(mean)
        test.insert(1, mean['test'][dataset_name])
        train.insert(1, mean['train'][dataset_name])
        labels.insert(1, 'combination')

    #print(test)
    #print(train)
    #print(legend)
    #print(labels)

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    if combination:
        color_list_test = [] # blue
        color_list_train = [] # orange

        for i in range(len(test)):
            if i == 1:
                color_list_test.append('tab:blue')
                color_list_train.append('tab:orange')

            else:
                color_list_test.append('lightsteelblue')
                color_list_train.append('bisque')

        rects2 = ax.bar(x - 2*width/len(legend), test, width, label='Test Acc', color = color_list_test)
        rects3 = ax.bar(x + 2*width/len(legend), train, width, label='Train Acc', color = color_list_train)

    else:
        rects2 = ax.bar(x - 2*width/len(legend), test, width, label='Test Acc')
        rects3 = ax.bar(x + 2*width/len(legend), train, width, label='Train Acc')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy Score')
    ax.set_ylim([0,100])
    ax.set_title(f'{model_type} model accuracy by techniques on {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='lightgrey')

    ax.plot([0, ax.get_xlim()[-1]] , [test[0], test[0]],
            ls='--', c='tab:blue', linewidth=1)
    ax.plot([0.2, ax.get_xlim()[-1]], [train[0], train[0]],
            ls='--', c='tab:orange', linewidth=1)

    plt.savefig(save_name)

def plot_bar_combination(file_content_list, save_name, model_type):
    labels = []
    legend = []
    for content in file_content_list:
        if content[1] not in legend:
            legend.append(content[1])

    techniques_test = {}
    techniques_train = {}

    for content in file_content_list:
        if content[1] not in labels:
            labels.append(content[1])
        if content[1] not in techniques_test:
            techniques_test[content[1]] = []
        if content[1] not in techniques_train:
            techniques_train[content[1]] = []
        techniques_test[content[1]].append(float(content[2])*100)
        techniques_train[content[1]].append(float(content[3])*100)

    mean = {}
    mean['test'] = []
    mean['train'] = []
    maxi = {}
    maxi['test'] = []
    maxi['train'] = []
    mini = {}
    mini['test'] = []
    mini['train'] = []

    for technique in labels:
    #labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        techniques_test[technique].sort()
        techniques_train[technique].sort()

        average_test = sum(techniques_test[technique]) / len(techniques_test[technique]) 
        mean['test'].append(average_test)
        maxi['test'].append(techniques_test[technique][len(techniques_test[technique]) - 1] - average_test)
        mini['test'].append(average_test - techniques_test[technique][0])

        average_train = sum(techniques_train[technique]) / len(techniques_train[technique]) 
        mean['train'].append(average_train)
        maxi['train'].append(techniques_train[technique][len(techniques_train[technique]) - 1] - average_train)
        mini['train'].append(average_train - techniques_train[technique][0])

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 2*width/len(legend), mean['test'], width, yerr=[mini['test'],maxi['test']], label='Testing Score')
    rects2 = ax.bar(x + 2*width/len(legend), mean['train'], width, yerr=[mini['train'],maxi['train']], label='Training Score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy Score')
    ax.set_ylim([0,100])
    ax.set_title(f'{model_type} model accuracy by multiple dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='lightgrey')

    (_, caps, _) = plt.errorbar(
        x- 2*width/len(legend), mean['test'], yerr=[mini['test'],maxi['test']], 
        fmt='o', markersize=3, capsize=8, c = 'black')
    for cap in caps:
        cap.set_markeredgewidth(1)
    (_, caps, _) = plt.errorbar(
        x+ 2*width/len(legend), mean['train'], yerr=[mini['train'],maxi['train']], 
        fmt='o', markersize=3, capsize=8, c = 'black')
    for cap in caps:
        cap.set_markeredgewidth(1)

    plt.savefig(save_name)

def lr_model():
    folder = 'LR_result.txt'
    file_content_list = readfile(folder)
    plot_bar_individual(file_content_list,'LR_Alzheimers_Handwriting', 'Alzheimers Handwriting', 'Binomial Logistic Regression')
    plot_bar_individual(file_content_list,'LR_Breast_Cancer', 'Breast Cancer', 'Binomial Logistic Regression')
    plot_bar_individual(file_content_list,'LR_Spam_Email', 'Spam Email', 'Binomial Logistic Regression')
    plot_bar_individual(file_content_list,'LR_Water_Potability', 'Water Potability', 'Binomial Logistic Regression')

    
    folder_comb = 'LR_combination_selection.txt'
    result_list = readfile(folder_comb)
    plot_bar_combination(result_list, 'LR_Combination', 'Binomial Logistic Regression')

    
    plot_bar_individual(file_content_list,'LR_Alzheimers_Handwriting_w_combination', 'Alzheimers Handwriting', 'Binomial Logistic Regression', result_list)
    plot_bar_individual(file_content_list,'LR_Breast_Cancer_w_combination', 'Breast Cancer', 'Binomial Logistic Regression', result_list)
    plot_bar_individual(file_content_list,'LR_Spam_Email_w_combination', 'Spam Email', 'Binomial Logistic Regression', result_list)
    plot_bar_individual(file_content_list,'LR_Water_Potability_w_combination', 'Water Potability', 'Binomial Logistic Regression', result_list)

def mlr_model():
    folder = 'MLR_result.txt'
    file_content_list = readfile(folder)
    plot_bar_individual(file_content_list,'MLR_Alzheimers_Handwriting', 'Alzheimers Handwriting', 'Multinomial Logistic Regression')
    plot_bar_individual(file_content_list,'MLR_Breast_Cancer', 'Breast Cancer', 'Multinomial Logistic Regression')
    plot_bar_individual(file_content_list,'MLR_Spam_Email', 'Spam Email', 'Multinomial Logistic Regression')
    plot_bar_individual(file_content_list,'MLR_Water_Potability', 'Water Potability', 'Multinomial Logistic Regression')

    
    folder_comb = 'MLR_combination_selection.txt'
    result_list = readfile(folder_comb)
    plot_bar_combination(result_list, 'MLR_Combination', 'Multinomial Logistic Regression')

    
    plot_bar_individual(file_content_list,'MLR_Alzheimers_Handwriting_w_combination', 'Alzheimers Handwriting', 'Multinomial Logistic Regression', result_list)
    plot_bar_individual(file_content_list,'MLR_Breast_Cancer_w_combination', 'Breast Cancer', 'Multinomial Logistic Regression', result_list)
    plot_bar_individual(file_content_list,'MLR_Spam_Email_w_combination', 'Spam Email', 'Multinomial Logistic Regression', result_list)
    plot_bar_individual(file_content_list,'MLR_Water_Potability_w_combination', 'Water Potability', 'Multinomial Logistic Regression', result_list)


def svm_model():
    folder = 'SVM_result.txt'
    file_content_list = readfile(folder)
    plot_bar_individual(file_content_list,'SVM_Alzheimers_Handwriting', 'Alzheimers Handwriting', 'Support Vector Machines')
    plot_bar_individual(file_content_list,'SVM_Breast_Cancer', 'Breast Cancer', 'Support Vector Machines')
    plot_bar_individual(file_content_list,'SVM_Spam_Email', 'Spam Email', 'Support Vector Machines')
    plot_bar_individual(file_content_list,'SVM_Water_Potability', 'Water Potability', 'Support Vector Machines')

    
    folder_comb = 'SVM_combination.txt'
    result_list = readfile(folder_comb)
    plot_bar_combination(result_list, 'SVM_Combination', 'Support Vector Machines')

    
    plot_bar_individual(file_content_list,'SVM_Alzheimers_Handwriting_w_combination', 'Alzheimers Handwriting', 'Support Vector Machines', result_list)
    plot_bar_individual(file_content_list,'SVM_Breast_Cancer_w_combination', 'Breast Cancer', 'Support Vector Machines', result_list)
    plot_bar_individual(file_content_list,'SVM_Spam_Email_w_combination', 'Spam Email', 'Support Vector Machines', result_list)
    plot_bar_individual(file_content_list,'SVM_Water_Potability_w_combination', 'Water Potability', 'Support Vector Machines', result_list)

if __name__ == "__main__":
    '''
    use these keywords
    dataset:
        Alzheimers Handwriting
        Breast Cancer
        Spam Email
        Water Potability

    technique: 
        base case
        normalization
        poly kernelization
        rbf kernelization
        sigmoid kernelization
        feature selection
        feature reduction
        shuffling
        bagging
        boosting
        hyperparameter tuning
        data reconstruction
        combination
    '''
    lr_model()
    #mlr_model()
    #svm_model()