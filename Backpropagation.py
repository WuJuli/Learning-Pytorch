import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad = True


# now w is a tensor
def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


print("Predict before training:", 4, forward(4).item())

epoch_list = []
loss_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        loss_list.append(loss_val.detach().numpy())
        epoch_list.append(epoch)
        print("\tgrad:", x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data

        # reset the grad
        w.grad.data.zero_()

    print("progess:", epoch, loss_val.item())

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
print("predict after training", 4, forward(4).item())
