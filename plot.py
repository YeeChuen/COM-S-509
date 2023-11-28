import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def readfile(folder):
    with open(folder, 'r') as f:
        to_return = []
        file_content = f.readlines()
        for file in file_content:
            file = file.replace("\n","")
            to_return.append(file.split(","))
        return to_return

def plot_bar_individual(file_content_list, save_name, dataset_name, model_type, combination = None):
    labels = []
    legend = []
    for content in file_content_list:
        if content[1] not in legend:
            legend.append(content[1])

    techniques_test = {}
    techniques_train = {}

    for content in file_content_list:
        if content[0] not in labels:
            labels.append(content[0])
        if content[1] not in techniques_test:
            techniques_test[content[1]] = []
        if content[1] not in techniques_train:
            techniques_train[content[1]] = []
        techniques_test[content[1]].append(float(content[2])*100)
        techniques_train[content[1]].append(float(content[3])*100)

    #labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    test = techniques_test[dataset_name]
    train = techniques_train[dataset_name]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

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

    #print(mean)
    #print(maxi)
    #print(mini)

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
    folder = 'example_LR_result.txt'
    file_content_list = readfile(folder)
    plot_bar_individual(file_content_list,'LR_Alzheimers_Handwriting', 'Alzheimers Handwriting', 'Binomial Logistic Regression')
    plot_bar_individual(file_content_list,'LR_Breast Cancer', 'Breast Cancer', 'Binomial Logistic Regression')
    plot_bar_individual(file_content_list,'LR_Spam Email', 'Spam Email', 'Binomial Logistic Regression')
    plot_bar_individual(file_content_list,'LR_Water Potability', 'Water Potability', 'Binomial Logistic Regression')

    
    folder_comb = 'example_LR_combination.txt'
    file_content_list = readfile(folder_comb)
    plot_bar_combination(file_content_list, 'LR_Combination', 'Binomial Logistic Regression')