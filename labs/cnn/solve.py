import torch
import time
import datetime
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torchvision


SUB_DIR_PICS = "fashion"
DRAW_MAIN_PICTURE = True
DOWNLOAD_DATASETS = True
PRINT_PIC = True
TEACH = True
# TEACH = False
PRINT_ALL_PICS = True
# PRINT_ALL_PICS = False


# lr: [0.001, 0.01, 0.1, 1]
# bs: [1, 32, 128, 256]
# e: [1, 5, 20]

# bad
LEARNING_RATE = 0.1
BATCH_SIZE = 128
EPOCH = 20


# LEARNING_RATE = 0.1
# BATCH_SIZE = 128
# EPOCH = 20

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = torch.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.act2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return torch.nn.functional.softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

train_MNIST = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=DOWNLOAD_DATASETS,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_MNIST = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=DOWNLOAD_DATASETS,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE, shuffle=True)

train_FashionMNIST = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('./', train=True, download=DOWNLOAD_DATASETS,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_FashionMNIST = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('./', train=False, download=DOWNLOAD_DATASETS,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ])),
    batch_size=BATCH_SIZE, shuffle=True)


def train():
    net.train()
    for X_batch, y_batch in train_FashionMNIST:
        optimizer.zero_grad()
        y_pred = net(X_batch)
        loss_value = loss_function(y_pred, y_batch)
        loss_value.backward()
        optimizer.step()


def test():
    correct = 0
    net.eval()
    with torch.no_grad():
        for X_batch, y_patch in test_FashionMNIST:
            y_pred = net(X_batch)
            pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(y_patch.data.view_as(pred)).sum()
    return correct / len(test_FashionMNIST.dataset)


def testWithSafe(X_batch_el):
    net.eval()
    y_pred = 0
    with torch.no_grad():
        y_pred = net([X_batch_el])
    return y_pred[0]


if TEACH:
    print('Start', flush=True)
    print(datetime.datetime.now(), flush=True)
    print(f'Learning rate: {LEARNING_RATE}', flush=True)
    print(f'Batch size: {BATCH_SIZE}', flush=True)
    print(f'Total epoch: {EPOCH}', flush=True)
    start_time = time.time()
    printedTime = False
    for epoch in range(EPOCH):
        train()
        epoch_res = test()
        if not printedTime:
            avg = time.time() - start_time
            spent_on_epoch = "%0.0f" % avg
            until_end = (avg * (EPOCH - 1))
            print(f'Avg seconds spent on each epoch: {spent_on_epoch} sec', flush=True)
            print(f'Until the end: {"%0.0f" % (until_end // 60)} min, {"%0.0f" % (until_end % 60)} sec', flush=True)
            printedTime = True
        print(f'accuracy on {epoch + 1}/{EPOCH} epoch: {epoch_res.item()}', flush=True)

matrix = [[0 for j in range(10)] for i in range(10)]
matrixPics = [[None for j in range(10)] for i in range(10)]
if PRINT_PIC:
    os.chdir(os.path.dirname(__file__))
    workingDirectory = os.getcwd()
    pictureFolder = os.path.join(workingDirectory, f'results/pictures/{SUB_DIR_PICS}')
    ind = 0

    net.eval()
    with torch.no_grad():
        for X_batch, y_patch in test_FashionMNIST:
            y_pred = net(X_batch)
            pred = y_pred.data.max(1, keepdim=True)[1]
            for j in range(len(X_batch)):
                x_batch_el = X_batch[j]
                y_real = y_patch.data.view_as(pred)[j].item()
                y_predicted = y_pred.data.max(1, keepdim=True)[1][j].item()
                matrix[y_real][y_predicted] += 1

                if matrix[y_real][y_predicted] == 1:
                    matrixPics[y_real][y_predicted] = X_batch[j][0]

                if PRINT_ALL_PICS:
                    if y_real != y_predicted:
                        pictureFolderByReal = os.path.join(pictureFolder, f'{y_real}')
                        if not os.path.exists(pictureFolderByReal):
                            os.makedirs(pictureFolderByReal)
                        pic_location = os.path.join(pictureFolderByReal, f'{y_predicted}_pred_|_i{ind}.png')
                        plt.imshow(x_batch_el[0], cmap='gray')
                        plt.title(f'real {y_real}, predicted {y_predicted}')
                        print(f'Saving {pic_location}', flush=True)
                        plt.savefig(pic_location)
                        ind += 1
    print('Matrix:')
    print(np.array(matrix))

if DRAW_MAIN_PICTURE:
    plt.figure()
    for i in range(10):
        for j in range(10):
            plt.subplot(len(matrixPics), len(matrixPics), i * 10 + j + 1)
            if matrixPics[i][j] is not None:
                plt.imshow(matrixPics[i][j], cmap='gray')
            plt.xticks([])
            plt.yticks([])
    os.chdir(os.path.dirname(__file__))
    workingDirectory = os.getcwd()
    pictureDir = os.path.join(workingDirectory, f'results/pictures/{SUB_DIR_PICS}.png')
    print(f'Saving global picture {pictureDir}', flush=True)
    plt.savefig(pictureDir)
