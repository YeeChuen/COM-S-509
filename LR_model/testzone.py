import numpy as np


y_pred = np.array([
    [0.8,0.4,0.5], # 0
    [0.9,0.7,0.6], # 0
    [float("nan"),float("nan"),float("nan")], # 2
    [0.9,0.4,0.4], # 0
    [0.8,0.9,0.5], # 1
    [0.9,0.7,0.6], # 0
    [0.1,0.7,0.6], # 1
    [0.9,0.4,0.99], # 2
    ])

max_row = []
for y_row in y_pred:
    idx_max = 0
    print(y_row)
    for i in range(len(y_row)):
        if y_row[i] >= y_row[idx_max]:
            idx_max = i
    max_row.append(idx_max)
for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
        if j == max_row[i]:
            y_pred[i][j] = int(1)
        else:
            y_pred[i][j] = int(0)

print(y_pred)