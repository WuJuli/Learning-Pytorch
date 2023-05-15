import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0])
w1.requires_grad = True

w2 = torch.tensor([1.0])
w2.requires_grad = True

b = torch.tensor([1.0])
b.requires_grad = True


# now w is a tensor, now this is new linear function w1x1^2 + w2x + b
def forward(x):
    return x * x * w1 + x * w2 + b


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
        print("\tgrad:", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        # reset the grad
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print("progess:", epoch, loss_val.item())

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
print("predict after training", 4, forward(4).item())
