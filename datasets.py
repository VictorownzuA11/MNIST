#Python files
import os.path
from PIL import Image
import cv2
import numpy as np
from glob import glob

# Torch files
import torch
import torch.utils.data
import torchvision.transforms as transforms

# ███    ███ ███    ██ ██ ███████ ████████ 
# ████  ████ ████   ██ ██ ██         ██    
# ██ ████ ██ ██ ██  ██ ██ ███████    ██    
# ██  ██  ██ ██  ██ ██ ██      ██    ██    
# ██      ██ ██   ████ ██ ███████    ██  

class MNISTDataset(torch.utils.data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """

    def __init__(self, train = True, seq_len = 1, display = 0, img_class = None, longdataset = 0):
        super(MNISTDataset, self).__init__()

        self.train = train  # training set or test set
        self.seq_len = seq_len # sequence length to return
        self.display = display # displays the returned sequence images
        self.img_class = img_class # values of images to return
        self.longdataset = longdataset # use long dataset (i.e. don't divide by seq_len length)

        if self.train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
        
        self.transform = transforms.ToTensor()

        self.data, self.targets = torch.load(os.path.join("./data", data_file))

        # Return only similar images (0, 1, 2, etc)
        if self.img_class is not None:
            data = [[], [], [], [], [], [], [], [], [], []]

            for i in range(len(self.targets)):
                data[self.targets[i]].append(self.data[i])

            self.data = data[self.img_class]

    def __getitem__(self, index):
        # Uses the entire dataset with repeats for training
        if not self.longdataset:
            index = index*self.seq_len

        # Get all the images in the sequence length
        imgs = torch.cat([self.transform(Image.fromarray(self.data[index + i].numpy(), mode='L')) \
            for i in range(self.seq_len)], axis=0)

        # Debug only, displays the returned images
        if self.display:
            imgview = np.concatenate([img.view(28, 28, 1) for img in imgs], axis=1)
            cv2.imshow("Item", imgview)
            cv2.waitKey(0)

        labels = torch.zeros([self.seq_len], dtype=torch.long)
        
        for i in range(self.seq_len):
            if self.img_class is not None:
                labels[i] = self.img_class
            else:
                labels[i] = self.targets[index]

        return imgs, labels

    def __len__(self):
        if self.longdataset:
            return len(self.data) - self.seq_len
        else:
            return (len(self.data) - self.seq_len) // self.seq_len

