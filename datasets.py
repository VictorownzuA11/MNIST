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


class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, seq="00", seq_len=1, display=1, longdataset=0):
        super(KITTIDataset, self).__init__()

        # FIXME: normalize the images once and don't do it again
        self.transform = transforms.Compose([
            transforms.Resize((384, 1280)),
            transforms.CenterCrop((384, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(
                #FIXME: need to normalize for RGB in the test set
                #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
                mean=[127. / 255., 127. / 255., 127. / 255.],
                std=[1 / 255., 1 / 255., 1 / 255.]
            )
        ])

        self.seq = seq # which sequence of images to use [0-10]
        self.seq_len = seq_len # sequence length to return
        self.display = display # displays the returned sequence images
        self.longdataset = longdataset # use long dataset (i.e. don't divide by seq length)

        # Get and store all the image paths in the sequences
        self.images = sorted(glob(f"C:/Users/Victor/Desktop/Shit/Projects/ESP-VO/dataset/sequences/{self.seq}/image_2/*.png"))

        #self.targets = self.get_targets()
        self.data = self.images # self.get_data()

    def __len__(self):
        if self.longdataset:
            return len(self.data) - self.seq_len
        else:
            return (len(self.data) - self.seq_len) // self.seq_len

    def __getitem__(self, index):
        # Uses the entire dataset with repeats for training
        if not self.longdataset:
            index = index*self.seq_len

        # Image dimensions (1280, 384, 3) RGB
        img = torch.zeros([self.seq_len, 6, 384, 1280])

        # Pose dimensions (x, y, z)
        label = torch.zeros([self.seq_len, 3])

        # Compase the images for the sequence length
        for i in range(self.seq_len):
            # Get the labels
            #label[i, :] = self.poses[index + i + 1] - self.poses[index + i]

            # Compose the image
            img[i, 0:3] = self.transform(Image.open(self.images[index + i]).convert('RGB'))
            img[i, 3:6] = self.transform(Image.open(self.images[index + i + 1]).convert('RGB'))

            # Debug only, displays the returned images
            if self.display:
                imgview = cv2.imread(self.images[index + i]) # Returns a np.array of size (376, 1241, 3)
                #print(np.shape(imgview))
                #img = self.transform(Image.fromarray(imgview)) # Returns a tensor of size (3, 384, 1280)
                #print(img.size())
                cv2.imshow("Item", imgview)
                cv2.waitKey(10)

        return img, label

    def get_targets(self):
        return

    def get_data(self):
        return


if __name__ == "__main__":
    test_dataset = KITTIDataset(display=1, seq_len=10)

    img, label = test_dataset.__getitem__(0)
