import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(reduction="sum")
optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer_Adagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)
optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_Adamax = torch.optim.Adamax(model.parameters(), lr=0.01)
optimizer_ASGD = torch.optim.ASGD(model.parameters(), lr=0.01)
optimizer_LBFGS = torch.optim.LBFGS(model.parameters(), lr=0.01)
optimizer_RMSprop = torch.optim.RMSprop(model.parameters(), lr=0.01)
optimizer_Rprop = torch.optim.Rprop(model.parameters(), lr=0.01)


def SGD():
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer_SGD.zero_grad()
        loss.backward()
        optimizer_SGD.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


def Adagrad():
    # when the epoch number becomes larger, the result does not behave better
    for epoch in range(40000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer_Adagrad.zero_grad()
        loss.backward()
        optimizer_Adagrad.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


def Adam():
    # when epoch equals 4000, this is near convergence
    for epoch in range(4000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer_Adam.zero_grad()
        loss.backward()
        optimizer_Adam.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


def Adamax():
    # when the epoch number equals 4000, the result has already convergence
    for epoch in range(4000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer_Adamax.zero_grad()
        loss.backward()
        optimizer_Adamax.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


def ASGD():
    # easily get convergence
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer_ASGD.zero_grad()
        loss.backward()
        optimizer_ASGD.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


def LBFGS():
    # LBFGS need function closure, and the behaviour of this algorithm is good
    for epoch in range(200):
        def closure():
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            print(epoch, loss.item())

            optimizer_LBFGS.zero_grad()
            loss.backward()
            return loss
        optimizer_LBFGS.step(closure)

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


def RMSprop():
    # easily get convergence
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer_RMSprop.zero_grad()
        loss.backward()
        optimizer_RMSprop.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


def Rprop():
    # the algorithm which needs the least epoch to get convergence
    for epoch in range(200):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer_Rprop.zero_grad()
        loss.backward()
        optimizer_Rprop.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)


if __name__ == '__main__':
    # SGD()
    # Adagrad()
    # Adam()
    # Adamax()
    # ASGD()
    LBFGS()
    # RMSprop()
    # Rprop()