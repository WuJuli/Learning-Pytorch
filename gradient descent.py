import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) * (y_pred - y)
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print("Predict before training:", 4, forward(4))
epoch_list = []
loss = []
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    epoch_list.append(epoch)
    loss.append(cost_val)
    print("epoch:", epoch, "w=", w, "loss=", cost_val)

plt.plot(epoch_list, loss)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
print("predict after training", 4, forward(4))
