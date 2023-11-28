import numpy as np
import matplotlib.pyplot as plt

'''
git remote set-url origin https://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@github.com/YeeChuen/COM-S-573.git
'''

def readfile(folder):
    with open(folder, 'r') as f:
        to_return = []
        file_content = f.readlines()
        for file in file_content:
            file = file.replace("\n","")
            to_return.append(file.split(","))

        return to_return

def plot_bar_individual(file_content_list, save_name):
    labels = []
    legend = []
    for content in file_content_list:
        if content[1] not in legend:
            legend.append(content[1])
    techniques = {}
    for content in file_content_list:
        if content[0] not in labels:
            labels.append(content[0])
        if content[1] not in techniques:
            techniques[content[1]] = []
        techniques[content[1]].append(float(content[2])*100)

    #labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    Alzheimers = techniques[legend[0]]
    Breast = techniques[legend[1]]
    Spam = techniques[legend[2]]
    Water = techniques[legend[3]]



    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 6*width/len(legend), Alzheimers, width, label=legend[0])
    rects2 = ax.bar(x - 2*width/len(legend), Breast, width, label=legend[1])
    rects3 = ax.bar(x + 2*width/len(legend), Spam, width, label=legend[2])
    rects4 = ax.bar(x + 6*width/len(legend), Water, width, label=legend[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0,100])
    ax.set_title('Model accuracy by techniques and dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / len(legend), height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_name)

if __name__ == "__main__":
    folder = 'example_LR_result.txt'
    file_content_list = readfile(folder)
    plot_bar_individual(file_content_list,'test')