import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# using linear model y = x * w + b
def forward(x):
    return x * w[index1] + b[index2]


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


mse_list = [[0 for j in range(41)] for i in range(41)]
w = np.arange(0.0, 4.1, 0.1)
b = np.arange(-2.0, 2.1, 0.1)

for index1 in range(len(w)):
    print("w=", w[index1])
    l_sum = 0
    for index2 in range(len(b)):
        print("b=", b[index2])
        for x_val, y_val in zip(x_data, y_data):
            y_pre_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
        mse_list[index1][index2] = l_sum / 3

W, B = np.meshgrid(w.tolist(), b.tolist())
plt.contourf(W, B, mse_list)
plt.colorbar()
plt.xlabel("w")
plt.ylabel("b")
plt.show()
