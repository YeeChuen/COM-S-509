import numpy as np
import matplotlib.pyplot as plt

def readfile(folder):
    with open(folder, 'r') as f:
        to_return = []
        file_content = f.readlines()
        for file in file_content:
            file = file.replace("\n","")
            to_return.append(file.split(","))

        return to_return

def plot_bar():
    data = [[30, 25, 50, 20],
    [40, 23, 51, 17],
    [35, 22, 45, 19]]
    X = np.arange(4)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
    ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

if __name__ == "__main__":
    folder = 'example_LR_result.txt'
    file_content_list = readfile(folder)

    print(file_content_list)