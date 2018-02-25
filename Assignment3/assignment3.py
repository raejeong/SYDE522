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
import pandas as pd

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

class CellTestDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx): 
        sample = {'image': self.X[idx,:,:,:]}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 17, padding=8)
        self.pool = nn.MaxPool2d((2,2))
        self.batchnorm1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 8, 17, padding=8)
        self.batchnorm2 = nn.BatchNorm2d(13)
        self.conv3 = nn.Conv2d(13, 13, 9, padding=4)
        self.batchnorm3 = nn.BatchNorm2d(26)
        self.conv4 = nn.Conv2d(15, 15, 9, padding=4)
        self.batchnorm4 = nn.BatchNorm2d(30)
        self.fc1 = nn.Linear(int(269568/32), 64)
        self.fc2 = nn.Linear(64, 20)
        self.dropout = nn.Dropout3d(p=0.05)

    def forward(self, x):
        x = functional.relu(self.batchnorm1(self.pool(torch.cat((self.conv1(x),x),1))))
        x = self.dropout(x)
        x = functional.relu(self.batchnorm2(self.pool(torch.cat((self.conv2(x),x),1))))
        x = self.dropout(x)
        x = functional.relu(self.batchnorm3(self.pool(torch.cat((self.conv3(x),x),1))))
        x = self.dropout(x)
        x = x.view(-1, int(269568/32))
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main(hp):
    # X, Y = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')
    testX, testY = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')
    trainX, trainY = np.load('moreX.npy'), np.load('moreY.npy')

    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(150), transforms.ToTensor()])

    # trainset = CellDataset(X[hp['num_test_images']:,:,:], Y[hp['num_test_images']:], transform)
    trainset = CellDataset(trainX, trainY, transform)

    trainset_loader = DataLoader(trainset, batch_size=hp['batch_size'], shuffle=True)

    # testset = CellDataset(X[:hp['num_test_images'],:,:,:], Y[:hp['num_test_images']], transform)
    testset = CellDataset(testX[:100,:,:,:], testY[:100], transform)

    testset_loader = DataLoader(testset, batch_size=hp['batch_size'], shuffle=True)

    net = Net()
    net.cuda()
    torch.save(net.state_dict(), "model.pt")
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

        if (100 * correct / total) > best_test_acc:
            best_test_acc = (100 * correct / total)
            torch.save(net.state_dict(), "model.pt")
        print('testAccuracy : %d %%' % (100 * correct / total))
        print('bestTestAccuracy : %d %%' % best_test_acc)

def test(hp):
    # X, Y = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')
    testX, testY = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')
    realTestX = np.expand_dims(np.load('X_test.npy'), 3)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(150), transforms.ToTensor()])

    # testset = CellDataset(X[:hp['num_test_images'],:,:,:], Y[:hp['num_test_images']], transform)
    testset = CellDataset(testX, testY, transform)
    testset_loader = DataLoader(testset, batch_size=hp['batch_size'], shuffle=True)
    realTestset = CellTestDataset(realTestX, transform)
    real_testset_loader = DataLoader(realTestset, batch_size=hp['batch_size'], shuffle=False)

    net = Net()
    net.cuda()
    net.load_state_dict(torch.load("model.pt"))
    dtype = torch.cuda.FloatTensor

    correct = 0
    total = 0
    output_data = {"Class":[]}
    for i, data in enumerate(testset_loader):
        inputs, labels = data['image'], data['label']
        # wrap them in Variable
        inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += torch.eq(predicted, labels.data).sum()
    
    print('testAccuracy : %d %%' % (100 * correct / total))

    correct = 0
    total = 0
    for i, data in enumerate(real_testset_loader):
        inputs = data['image']
        # wrap them in Variable
        inputs= Variable(inputs).type(dtype)

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for j in range(predicted.size(0)):
            output_data['Class'].append(int(predicted[j]))

    # for i in range(len(testY)):
    #     correct += int(testY[i]==output_data['Class'][i])
    #     total += 1

    # print('realTestAccuracy : %d %%' % (100 * correct / total))
    df = pd.DataFrame(output_data, columns=["Class"])
    df.to_csv('test_prediction.csv')

if __name__ == "__main__":
    hyperparameters = {
        "num_epoch":200,
        "batch_size":32,
        "num_test_images":100,
        "learning_rate":0.0001,
        "print_every":10,
    }
    test(hyperparameters)
    # main(hyperparameters)
