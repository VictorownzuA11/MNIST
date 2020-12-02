# Python Libraries
from time import process_time as ptime
import time
import argparse
import cv2
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchsummary import summary as netsummary

# Networks
from models import *

# Datasets
from datasets import *

if __name__ == "__main__":
    # Parse the input arguments of main.py
    parser = argparse.ArgumentParser(description='ESP-VO Model implemented in PyTorch')

    # General arguments
    parser.add_argument('--model', default="lstm", type=str, help='Chooses the model to be run')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images per batch')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train on')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='Learning rate (default: 0.1)')
    parser.add_argument('--seq_len', default=2, type=int, help='Length of the sequence of images used')

    parser.add_argument('--summary', default=1, type=int, help='Print a summary of the model used')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), \
        type=str, help='Device to run PyTorch with')

    parser.add_argument('--display', default=1, type=int, help='Displays the MNIST batch images')
    parser.add_argument('--wait', default=0, type=int, help='Waits after each image')
    parser.add_argument('--manual_seed', default=1, type=int, help='Set the manual seed for deterministic \
        results')

    args = parser.parse_args()


    '''
    STEP 0: INITIALIZATION
    '''
    # Display meta data about the argparse parameters
    print("\n==========================================================================================")
    print("Args".center(90))
    print("==========================================================================================")
    
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    print("==========================================================================================\n")

    if args.manual_seed:
        seed = 999
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    '''
    STEP 1: INSTANTIATE MODEL CLASS
    '''
    input_dim = 28
    hidden_dim = 100
    layer_dim = 3
    output_dim = 10
    img_dim_x = 28
    img_dim_y = 28

    # Number of steps to unroll
    seq_dim = 28
    seq_len = 1
    use_seq = 1

    modelname = args.model
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs
    
    if args.wait:
        wait_time = 0
    else:
        wait_time = 1

    print("\n==========================================================================================")
    print(f"Loading model {modelname}".center(90))
    print("==========================================================================================")

    if (modelname == "lstm"):
        # Model Structure
            # (100 x 28 x 28)
            # (batch_size, seq_dim, input_dim)
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    elif (modelname == "lstm2"):
        seq_len = args.seq_len
        seq_dim *= seq_len
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    elif (modelname == "lstmcell"):
        print("Warning, lstmcell does not converge")
        model = LSTMCellModel(input_dim, hidden_dim, layer_dim, output_dim)
    elif (modelname == "rnn"):
        model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    elif (modelname == "cnn"):
        use_seq = 0
        model = CNNModel()
    elif (modelname == "lenet"):
        use_seq = 0
        model = LENETModel()
    elif (modelname == "crnn"):
        seq_len = args.seq_len
        use_seq = 0
        model = CRNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    else:
        exit("Invalid Model")

    # Move model to device
    model.to(args.device)

    # Print a Summary of the model
    if args.summary:
        if use_seq:
            netsummary(model, (seq_dim, input_dim), device="cuda")
        else:
            netsummary(model, (1, seq_dim, input_dim), device="cuda")


    '''
    STEP 2: LOADING DATASET
    '''
    # Count of Images in Training
        # 0: 5923
        # 1: 6742
        # 2: 5958
        # 3: 6131
        # 4: 5842
        # 5: 5421
        # 6: 5918
        # 7: 6265
        # 8: 5851
        # 9: 5949
    train_dataset = torch.utils.data.ConcatDataset([MNISTDataset(train=True, seq_len=seq_len, \
        img_class=image, display=0) for image in range(10)])

    # Count of Images in Test
        # 0: 980
        # 1: 1135
        # 2: 1032
        # 3: 1010
        # 4: 982
        # 5: 892
        # 6: 958
        # 7: 1028
        # 8: 974
        # 9: 1009
    if seq_len > 1:
        test_dataset = torch.utils.data.ConcatDataset([MNISTDataset(train=False, seq_len=seq_len, \
            img_class=image, display=0) for image in range(10)])
    else:
        test_dataset = MNISTDataset(train=False, seq_len=seq_len)


    '''
    STEP 3: MAKING DATASET ITERABLE
    '''
    # Set the CUDA params
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False,
                                        **kwargs)


    '''
    STEP 4: INSTANTIATE LOSS CLASS
    '''
    criterion = nn.CrossEntropyLoss()


    '''
    STEP 5: INSTANTIATE OPTIMIZER CLASS
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    '''
    STEP 6: TRAIN THE MODEL
    '''
    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(train_loader):
            # Display the batch
            if args.display:
                # Generate the output of all images in the batch
                if seq_len > 1:
                    imgview = np.concatenate([np.concatenate([image.view(img_dim_x, img_dim_y, 1) for image in image_batch], axis= 1) for image_batch in images], axis=0)
                else:
                    imgview = np.concatenate([image.view(img_dim_x, img_dim_y, 1) for image in images], axis=1)
                
                # Display the image for 1ms
                cv2.imshow("Train Batch", imgview)
                cv2.waitKey(wait_time)

            # Load images as Variable
            if use_seq:
                # (batch_size, seq_dim, input_dim)
                images = images.view(-1, img_dim_x, img_dim_y).requires_grad_().to(device)
            else:
                # (batch_size, 1, seq_dim, input_dim)
                images = images.requires_grad_().to(device)

            # Reformat the labels from [batch_size, seq_len] to batch_size*seq_len]
            labels = labels.view(-1).to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Test the model halfway, and at the end of the batch
            if (batch+1) % (len(train_loader)//2) == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    # Display the batch
                    if args.display:
                        # Generate the output of all images in the batch
                        if seq_len > 1:
                            imgview = np.concatenate([np.concatenate([image.view(img_dim_x, img_dim_y, 1) for image in image_batch], axis= 1) for image_batch in images], axis=0)
                        else:
                            imgview = np.concatenate([image.view(img_dim_x, img_dim_y, 1) for image in images], axis=1)
                        
                        # Display the image for 1ms
                        cv2.imshow("Test Batch", imgview)
                        cv2.waitKey(wait_time)

                    # Format the images to the model input parameters
                    if use_seq:
                        images = images.view(-1, img_dim_x, img_dim_y).to(device)
                    else:
                        images = images.to(device)
                    
                    # Reformat the labels from [batch_size, seq_len] to batch_size*seq_len]
                    labels = labels.view(-1).to(device)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()

                accuracy = 100 * correct // total

                # Print Loss
                print(f"Epoch: {epoch+1}. Batch: {batch+1}. Loss: {loss.item()}. Accuracy: {accuracy}.")