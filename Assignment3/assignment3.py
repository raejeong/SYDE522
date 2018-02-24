#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable

class CellDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx): 
        sample = {'image': self.X[idx,:,:,:], 'label': self.Y[idx]}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(int(3072000/4), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 20)

    def forward(self, x):
        x = functional.relu(self.batchnorm1(self.conv1(x)))
        x = functional.relu(self.batchnorm2(self.conv2(x)))
        x = x.view(-1, int(3072000/4))
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    X, Y = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = CellDataset(X[100:,:,:], Y[100:], transform)
    trainset_loader = DataLoader(trainset, batch_size=4, shuffle=True)

    testset = CellDataset(X[:100,:,:,:], Y[:100], transform)
    testset_loader = DataLoader(testset, batch_size=4, shuffle=True)

    net = Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainset_loader):
            inputs, labels = data['image'], data['label']
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels.data).sum()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        print('Train Accuracy : %d %%' % (100 * correct / total))
        correct = 0
        total = 0
        for i, data in enumerate(testset_loader):
            inputs, labels = data['image'], data['label']
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels.data).sum()

        print('Accuracy : %d %%' % (100 * correct / total))




if __name__ == "__main__":
    main()