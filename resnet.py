'''
Author: Yumo Chi
Modified based on the work from the neural_networks_tutorial.py from pytorch tutorial
as well as work from yunjey's RNN tutorial, especially regarding treatment of 
shortcuting and use of filter block
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

import torchvision
import torchvision.transforms as transforms
# import torchsample as ts

import numpy as np
from random import randint
from torch.autograd import Variable

from torchvision import datasets
from torch.utils.data import DataLoader



# 2 filter block
class ResNet_Block(nn.Module):
    ''' constructor function
        param:
            in_channels: input channels
            out_channels: output channels
            stride: stride for filters in the block
            downsample: downsample function 
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNet_Block, self).__init__()
        # input image channel, output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
        # batch normalization
        self.conv_1_bn = nn.BatchNorm2d(out_channels)

        # input image channel, output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # batch normalization
        self.conv_2_bn = nn.BatchNorm2d(out_channels)

        # downsample flag
        self.downsample = downsample

    # forward function
    def forward(self, x):
        shortcut_input = x
        # F.relu(self.conv1_bn(self.conv1(x)))
        out = F.relu(self.conv_1_bn(self.conv_1(x)))
        out = self.conv_2_bn(self.conv_2(out))

        if self.downsample:
            shortcut_input = self.downsample(shortcut_input)

        out += shortcut_input
        return out

# ResNet - Residual Network
class ResNet(nn.Module):
    ''' constructor function
        param:
            block: blocks for the Residual Network
            lays: list to specify number of blocks in each layer
            num_classes: number of classes 
    '''
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 32
        # kernel
        # 3 input image channel, 32 output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # batch normalization
        # normalize conv 32 output channels
        self.conv1_bn = nn.BatchNorm2d(32)

        #drop out layers for conv
        self.conv1_dol = nn.Dropout(p=0.1)

        # conv2 block, 2 3x3 conv, 32 input, 32 output, 1 stride
        self.bl_2 = self.make_block_layer(block, 32, 1, layers[0])
        # conv2 block, 4 3x3 conv, 32 input, 64 output, 2 stride
        self.bl_3 = self.make_block_layer(block, 64, 2, layers[1])
        # conv2 block, 4 3x3 conv, 64 input, 128 output, 2 stride
        self.bl_4 = self.make_block_layer(block, 128, 2, layers[2])
        # conv2 block, 2 3x3 conv, 128 input, 256 output, 2 stride
        self.bl_5 = self.make_block_layer(block, 256, 2, layers[3])

        # 1 fully connected layer 
        self.fc1 = nn.Linear(1024, 100)

    # function to create layer based on number of blocks 
    def make_block_layer(self, block, out_channels, stride=1, blocks=2):
        downsample = None
        # decide if downsample is needed
        if (stride != 1) or (self.in_channels != out_channels):
            # use conv2d to subsample if needed
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        filters = []
        filters.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            filters.append(block(out_channels, out_channels))
        # connect the filters 
        return nn.Sequential(*filters)
    # function to help reduce input to a 1D tensor 
    def num_flat_features(self, x):
        size = x[0].size() # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # apply dropout relu batch normalization after applying 1st conv
        x = self.conv1_dol(F.relu(self.conv1_bn(self.conv1(x))))
        # apply block layer 2
        x = self.bl_2(x)
        # apply block layer 3 
        x = self.bl_3(x) 
        # apply block layer 4
        x = self.bl_4(x)
        # apply max_pool and block layer 5 
        x = F.max_pool2d(self.bl_5(x), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        # apply first fully connected layer
        x = self.fc1(x)
        # apply soft_max to the result
        return F.log_softmax(x, dim=1)

