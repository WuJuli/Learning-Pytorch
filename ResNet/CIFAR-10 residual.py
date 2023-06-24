import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.nn.functional as F

# prepare dataset
batch_size = 64

# Define the data augmentation transformations
train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='../dataset/CIFAR10/', train=True, download=False, transform=train_transform)
test_dataset = datasets.CIFAR10(root='../dataset/CIFAR10/', train=False, download=False, transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv(x)
        y = F.relu(self.bn(y))
        y = self.conv(y)
        y = F.relu(self.bn(y))
        return F.relu(x + y)


class Inception20(nn.Module):
    def __init__(self):
        super(Inception20, self).__init__()
        # CIFAR-10 input size is 32, input channel is RGB = 3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.rblock3 = ResidualBlock(64)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # output size is 16

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # output size is 8
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Linear(1024, 10)


        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        # 1
        x = self.rblock1(x)
        # 2
        x = self.rblock1(x)
        # 3
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        # 1
        x = self.rblock2(x)
        # 2
        x = self.rblock2(x)
        # 3
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        # 1
        x = self.rblock3(x)
        # 2
        x = self.rblock3(x)
        # 3
        x = self.rblock3(x)

        x = self.global_avg_pool(x)
        # print(x.shape)

        x = x.view(in_size, -1)
        x = self.fc(x)


        return x


model = Inception20()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()
