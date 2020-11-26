import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Note: ASCI generator: http://www.patorjk.com/software/taag/#p=display&v=3&f=ANSI%20Regular&t

# ██      ███████ ████████ ███    ███ 
# ██      ██         ██    ████  ████ 
# ██      ███████    ██    ██ ████ ██ 
# ██           ██    ██    ██  ██  ██ 
# ███████ ███████    ██    ██      ██

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # NOTE: Scaling the imput allows it to train much faster
        x *= 255

        # Get dimensions of input
        batch_size = x.size(0)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().to(device)

        # One time step
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> batch_size, seq_dim, hidden_size
        # out[:, -1, :] --> batch_size, hidden_size --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> batch_size, output_size
        return out

# ██      ███████ ████████ ███    ███      ██████ ███████ ██      ██      
# ██      ██         ██    ████  ████     ██      ██      ██      ██      
# ██      ███████    ██    ██ ████ ██     ██      █████   ██      ██      
# ██           ██    ██    ██  ██  ██     ██      ██      ██      ██      
# ███████ ███████    ██    ██      ██      ██████ ███████ ███████ ███████

class LSTMCellModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMCellModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstmcell1 = nn.LSTMCell(input_dim, hidden_dim)
        self.lstmcell2 = nn.LSTMCell(input_dim, hidden_dim)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # NOTE: Scaling the imput allows it to train much faster
        x *= 255

        # Get dimensions of input
        batch_size = x.size(0)
        seq_dim = x.size(1)

        # Initialize hidden state with zeros
        hx = torch.zeros(seq_dim, self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        cx = torch.zeros(seq_dim, self.hidden_dim).requires_grad_().to(device)

        # One time step
        out = torch.zeros([batch_size, seq_dim, self.hidden_dim]).requires_grad_().to(device)
        for i in range(batch_size):
            hx, cx = self.lstmcell1(x[i], (hx.detach(), cx.detach()))
            hx, cx = self.lstmcell2(x[i], (hx.detach(), cx.detach()))
            out[i] = hx

        # Index hidden state of last time step
        # out.size() --> batch_size, seq_dim, hidden_size
        # out[:, -1, :] --> batch_size, hidden_size --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> batch_size, output_size
        return out

# ██████  ███    ██ ███    ██ 
# ██   ██ ████   ██ ████   ██ 
# ██████  ██ ██  ██ ██ ██  ██ 
# ██   ██ ██  ██ ██ ██  ██ ██ 
# ██   ██ ██   ████ ██   ████

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # One time step
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

#  ██████ ███    ██ ███    ██ 
# ██      ████   ██ ████   ██ 
# ██      ██ ██  ██ ██ ██  ██ 
# ██      ██  ██ ██ ██  ██ ██ 
#  ██████ ██   ████ ██   ████ 

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2 
        out = self.maxpool2(out)

        # Resize
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

# ██      ███████ ███    ██ ███████ ████████ 
# ██      ██      ████   ██ ██         ██    
# ██      █████   ██ ██  ██ █████      ██    
# ██      ██      ██  ██ ██ ██         ██    
# ███████ ███████ ██   ████ ███████    ██   

class LENETModel(nn.Module):
    def __init__(self):
        super(LENETModel, self).__init__()

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

#  ██████ ██████  ███    ██ ███    ██ 
# ██      ██   ██ ████   ██ ████   ██ 
# ██      ██████  ██ ██  ██ ██ ██  ██ 
# ██      ██   ██ ██  ██ ██ ██  ██ ██ 
#  ██████ ██   ██ ██   ████ ██   ████

class CRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(CRNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

         # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(32*4*4, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2 
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 4, 4)
        print(out.size())
        # out.size(0): 100
        # New out size: (10, 10, 32*4*4)
        out = out.view(10, 10, -1)
        print(out.size())

        h0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_().to(device)

        # One time step
        out, (hn, cn) = self.lstm(out, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out


# ███████ ███████ ██████        ██    ██  ██████  
# ██      ██      ██   ██       ██    ██ ██    ██ 
# █████   ███████ ██████  █████ ██    ██ ██    ██ 
# ██           ██ ██             ██  ██  ██    ██ 
# ███████ ███████ ██              ████    ██████ 



#  ██████   █████  ███    ██ 
# ██       ██   ██ ████   ██ 
# ██   ███ ███████ ██ ██  ██ 
# ██    ██ ██   ██ ██  ██ ██ 
#  ██████  ██   ██ ██   ████