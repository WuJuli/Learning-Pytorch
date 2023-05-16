import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


# prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


def train(epoch):
    train_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()

        train_loss += loss.item()
        count = i

    if epoch % 2000 == 1999:
        print("train loss: ", train_loss / count, end=',')


def test():
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        acc = torch.eq(y_pred_label, y_test).sum().item() / y_test.size(0)
        print("test acc :", acc)


dataset = np.loadtxt('/home/share449/PycharmProjects/dataset/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = dataset[:, :-1]
y_data = dataset[:, [-1]]
X_trian, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=322)

train_dataset = DiabetesDataset(X_trian, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(50000):
        train(epoch)
        if epoch % 2000 == 1999:
            test()
