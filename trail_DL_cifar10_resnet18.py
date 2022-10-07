#%%
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from models import *
import pandas as pd
import datetime
from utils import data_save, model_save



#%%
trainlist = []
validlist = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr = 0.01

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 40 * 40
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data/CIFAR10', train=True, download=True, transform=transform_train)
length = len(trainset)
train_size, valid_size = int(0.8*length), int(0.2*length)
train_set,valid_set=torch.utils.data.random_split(trainset,[train_size,valid_size])
trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True, num_workers=4)
validloader = torch.utils.data.DataLoader(
    valid_set, batch_size=128, shuffle=False, num_workers=4)

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)  # DataParallel
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_accuracy = 100. * correct / total
    trainlist.append([epoch, (train_loss / (batch_idx + 1)), train_accuracy])


def valid(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(validloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = 100. * correct / total
    validlist.append([epoch, (test_loss/(batch_idx+1)),valid_accuracy])
    # Save checkpoint.
    if valid_accuracy > best_acc:
        print('Saving..')
        torch.save(net.state_dict(), 'result_temp.pt')
        best_acc = valid_accuracy


#%%
for epoch in range(start_epoch, start_epoch+2):
    train(epoch)
    valid(epoch)
    print(best_acc)

#%%
nowTime = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
model_save(nowTime, best_acc)
data_save(trainlist,validlist,nowTime,best_acc)
print(best_acc)
