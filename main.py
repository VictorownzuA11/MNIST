# Python Libraries
from time import process_time as ptime
import argparse

# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchsummary import summary

# Networks
from models import *

if __name__ == "__main__":
    # Parse the input arguments of main.py
    parser = argparse.ArgumentParser(description='ESP-VO Model implemented in PyTorch')

    # General arguments
    parser.add_argument('--model', default="lstm", type=str, help='Chooses the model to be run')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images per batch')
    parser.add_argument('--iter', default=6000, type=int, help='Number of iterations to train on')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), \
        type=str, help='Device to run PyTorch with')

    parser.add_argument('--progress', default=1, type=int, help='Shows progress bar for training')
    parser.add_argument('--manual_seed', default=0, type=int, help='Set the manual seed for deterministic results')

    args = parser.parse_args()


    '''
    STEP 1: LOADING DATASET
    '''
    train_dataset = dsets.MNIST(root='./', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=False)

    test_dataset = dsets.MNIST(root='./', 
                            train=False, 
                            transform=transforms.ToTensor())

    '''
    STEP 2: MAKING DATASET ITERABLE
    '''
    batch_size = args.batch_size
    n_iters = args.iter
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)


    '''
    STEP 3: INSTANTIATE MODEL CLASS
    '''
    input_dim = 28
    hidden_dim = 100
    layer_dim = 3  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
    output_dim = 10

    modelname = args.model

    print("\n==========================================================================================")
    print(f"Loading model {modelname}".center(90))
    print("==========================================================================================")

    if (modelname == "lstm"):
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
        use_seq = 1
    elif (modelname == "lstmcell"):
        model = LSTMCellModel(input_dim, hidden_dim, layer_dim, output_dim)
        use_seq = 0
    elif (modelname == "rnn"):
        model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
        use_seq = 1
    elif (modelname == "cnn"):
        model = CNNModel()
        use_seq = 0
    elif (modelname == "lenet"):
        model = lenet()
        use_seq = 0

    # Move model to device
    model.to(args.device)

    # Print a Summary of the model
    if args.summary:
        summary(model, (28, 28), device="cuda")

    '''
    STEP 4: INSTANTIATE LOSS CLASS
    '''
    criterion = nn.CrossEntropyLoss()

    '''
    STEP 5: INSTANTIATE OPTIMIZER CLASS
    '''
    learning_rate = 0.1
    #learning_rate = 0.01

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    '''
    STEP 6: TRAIN THE MODEL
    '''

    # Number of steps to unroll
    seq_dim = 28

    iter = 0
    print(num_epochs)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as Variable
            # (batch_size, seq_dim, input_dim)
            # (100 x 1 x 28 x 28)
            if use_seq:
                images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            else:
                images = images.requires_grad_().to(device)
            # (100 x 28 x 28)

            labels = labels.to(device)

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

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    if use_seq:
                        images = images.view(-1, seq_dim, input_dim).to(device)
                    else:
                        images = images.to(device)
                    
                    labels = labels.to(device)

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
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))