import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as F

import struct as st
import numpy as np

from helper import *

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3, padding=(2,2))
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MNISTDataset(td.Dataset):
    def __init__(self, train_labels, train_images):
        self.labels = self.getLabels(train_labels)
        self.images = self.getImages(train_images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label, index

    def getLabels(self, filename):
        print('Getting labels')
        with open(filename,'rb') as f:
            chunk = f.read(8)
            #MSB = st.unpack('>I',chunk[0:4])[0]
            labels = st.unpack('>I',chunk[4:8])[0]
            labels_l = torch.zeros([labels, 10])
            chunk = f.read(labels)

            for i in range(labels):
                index =  st.unpack('B', chunk[i:i+1])[0]
                labels_l[i, index] = 1

        print('Done getting labels')
        return labels_l

    def getImages(self, filename):
        print('Getting images')
        with open(filename, 'rb') as f:
            chunk = f.read(16)
            #MSB = st.unpack('>I', chunk[0:4])[0]
            images = st.unpack('>I', chunk[4:8])[0]
            xDim = st.unpack('>I', chunk[8:12])[0]
            yDim = st.unpack('>I', chunk[12:16])[0]
            imageSize = xDim * yDim
            temp = np.zeros([imageSize])
            images_l = torch.zeros([images, 1, xDim, yDim])
            chunk = f.read(imageSize)

            for i in range(images):
                temp[:] = st.unpack('=784B', chunk)

                images_l[i] = torch.from_numpy(temp.reshape([1, xDim, yDim]))
                chunk = f.read(imageSize)

        print('Done getting images')
        return images_l