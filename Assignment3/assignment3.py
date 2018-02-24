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
import pdb

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
        self.conv1 = nn.Conv2d(1, 5, 9, padding=4)
        self.pool = nn.MaxPool2d((2,2))
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 8, 9, padding=4)
        self.batchnorm2 = nn.BatchNorm2d(14)
        self.conv3 = nn.Conv2d(14, 20, 9, padding=4)
        self.batchnorm3 = nn.BatchNorm2d(34)
        self.fc1 = nn.Linear(int(352512/32), 64)
        self.fc2 = nn.Linear(64, 20)
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        x = functional.relu(self.batchnorm1(self.pool(torch.cat((self.conv1(x),x),1))))
        x = self.dropout(x)
        x = functional.relu(self.batchnorm2(self.pool(torch.cat((self.conv2(x),x),1))))
        x = self.dropout(x)
        x = functional.relu(self.batchnorm3(self.pool(torch.cat((self.conv3(x),x),1))))
        x = self.dropout(x)
        x = x.view(-1, int(352512/32))
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main(hp):
    # X, Y = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')
    testX, testY = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')
    trainX, trainY = np.load('moreX.npy'), np.load('moreY.npy')

    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(150), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = CellDataset(X[hp['num_test_images']:,:,:], Y[hp['num_test_images']:], transform)
    trainset = CellDataset(trainX, trainY, transform)

    trainset_loader = DataLoader(trainset, batch_size=hp['batch_size'], shuffle=True)

    # testset = CellDataset(X[:hp['num_test_images'],:,:,:], Y[:hp['num_test_images']], transform)
    testset = CellDataset(testX[:100,:,:,:], testY[:100], transform)

    testset_loader = DataLoader(testset, batch_size=hp['batch_size'], shuffle=True)

    net = Net()
    net.cuda()
    dtype = torch.cuda.FloatTensor
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=hp['learning_rate'])

    best_test_acc = 0
    for epoch in range(hp['num_epoch']):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainset_loader):
            inputs, labels = data['image'], data['label']
            # wrap them in Variable
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

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
            if i % hp['print_every'] == hp['print_every']-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / hp['print_every']))
                running_loss = 0.0
        print('Train Accuracy : %d %%' % (100 * correct / total))
        correct = 0
        total = 0
        for i, data in enumerate(testset_loader):
            inputs, labels = data['image'], data['label']
            # wrap them in Variable
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels.data).sum()

        best_test_acc = max((100 * correct / total), best_test_acc)
        print('testAccuracy : %d %%' % (100 * correct / total))
        print('bestTestAccuracy : %d %%' % best_test_acc)

if __name__ == "__main__":
    hyperparameters = {
        "num_epoch":50,
        "batch_size":32,
        "num_test_images":100,
        "learning_rate":0.0005,
        "print_every":10,
    }
    main(hyperparameters)