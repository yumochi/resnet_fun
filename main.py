'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
import time


def main():
    # set up argparser
    parser = argparse.ArgumentParser(description='hw3')
    # batch-size
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    # epochs
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    # learning rate
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    # gpu setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args = parser.parse_args()

    # test if gpu should be used
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # adding in data augmentation transformations
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # just transform to tensor for test_data
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # data loader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=1)
    batch_size = args.batch_size

    # initialize the residual net
    model = ResNet(ResNet_Block,[2,2,2,2]).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    model.train()

    current_time = time.time()
    for epoch in range(1, args.epochs + 1):
        #Randomly shuffle data every epoch
        train_accu = []
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data), Variable(target)

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = F.nll_loss(output, target)

            # start backprop
            loss.backward()
            # optimize
            optimizer.step()

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
               )*100.0
            train_accu.append(accuracy)

        accuracy_epoch = np.mean(train_accu)
        time_used = time.time() - current_time
        current_time = time.time()
        print(epoch, accuracy_epoch, time_used)
    
    model.eval()
    test_accu = []
    for batch_idx, (data, target) in enumerate(test_loader, 0):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
        test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
    print(accuracy_test)


if __name__ == '__main__':
    main()