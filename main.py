import torch
import torch.utils.data as td
import torch.optim as optim

from mnist import *

# Set the device for torch as GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = MNIST()
    model.to(device)

    batch_size = 64

    train_loader = td.DataLoader(MNISTDataset('train-labels.idx1-ubyte', 'train-images.idx3-ubyte'), \
        batch_size=batch_size, shuffle=True, drop_last=True)

    # Set the model to train
    model.train()
    model.training = True

    # Set the loss/criterion function
    criterion = torch.nn.MSELoss()

    # Set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)) #lr = 0.001, B1 = 0.9 B1 = 0.999 Epoch = 200, Dropout, early stopping (see: https://stackoverflow.com/questions/2976452/whats-is-the-difference-between-train-validation-and-test-set-in-neural-netwo), and incremental training techniques are introduced, uses pretrained FlowNet model CNN (Dosovitskiy 2015)    

    # Loop through all the training data
    for batch_idx, (imgs, labels, index) in enumerate(train_loader):
        optimizer.zero_grad()   # zero the gradient buffers
        
        # Print data variables
        # print("Images:", imgs.shape)
        # print("Labels:", labels.shape)
        # print("Batch indexes:", index)

        # Evaluate the batch
        est_labels = model(imgs.to(device))

        # Print results
        # print(est_labels)
        # print(labels)

        # Calculate the loss of the batch
        loss = criterion(est_labels, labels.to(device))

        # Compute gradient and do  optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        print("loss:", loss.item())