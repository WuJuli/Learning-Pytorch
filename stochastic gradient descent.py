import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


def gradient(x, y):
    return 2 * x * (x * w - y)


print("Predict before training:", 4, forward(4))
epoch_list = []
loss_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad_val = gradient(x, y)
        w -= 0.01 * grad_val
        print("\tgrad:", x, y, grad_val)
        loss_val = loss(x, y)
        loss_list.append(loss_val)
        epoch_list.append(epoch)
    print("epoch:", epoch, "w=", w, "loss=", loss_val)

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
print("predict after training", 4, forward(4))
