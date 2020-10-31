#Python files
import os.path
from PIL import Image
import cv2
import numpy as np

# Torch files
import torch
import torch.utils.data
import torchvision.transforms as transforms


class MNISTDataset(torch.utils.data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """

    def __init__(self, train = True):
        super(MNISTDataset, self).__init__()

        self.train = train  # training set or test set

        if self.train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
        
        self.transform = transforms.ToTensor()

        self.data, self.targets = torch.load(os.path.join("./data", data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # Convert to PIL image and normalize to 1
        img = self.transform(Image.fromarray(img.numpy(), mode='L'))

        return img, target

    def __len__(self):
        return len(self.data)


class MNISTDataset2(torch.utils.data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """

    def __init__(self, train = True, seq = 1, display = 0, images = None, longdataset = 1):
        super(MNISTDataset2, self).__init__()

        self.train = train  # training set or test set
        self.seq = seq # sequence length to return
        self.display = display # displays the returned sequence images
        self.images = images # values of images to return
        self.longdataset = longdataset # use long dataset (i.e. don't divide by seq length)

        if self.train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
        
        self.transform = transforms.ToTensor()

        self.data, self.targets = torch.load(os.path.join("./data", data_file))

        # Return only similar images (0, 1, 2, etc)
        if self.images is not None:
            data = [[], [], [], [], [], [], [], [], [], []]

            for i in range(len(self.targets)):
                data[self.targets[i]].append(self.data[i])

            self.data = data[self.images]

    def __getitem__(self, index):
        # Image dimensions (1280, 384, 3) RGB
        imgs = torch.zeros([self.seq, 1, 28, 28])

        if not self.longdataset:
            index = index*self.seq

        # Compase the images for the sequence length
        for i in range(self.seq):
            img = self.data[index + i]

            # Convert to PIL image and normalize to 1
            imgs[i] = self.transform(Image.fromarray(img.numpy(), mode='L'))

        # Debug only, displays the returned images
        if self.display:
            imgview = np.concatenate([image.view(28, 28, 1) for image in imgs], axis=1)
            cv2.imshow("Batch", imgview)
            cv2.waitKey(0)

        if self.images is not None:
            return imgs, self.images
        else:
            return imgs, self.targets[index]

    def __len__(self):
        if self.longdataset:
            return len(self.data) - self.seq
        else:
            return (len(self.data) - self.seq) // self.seq
