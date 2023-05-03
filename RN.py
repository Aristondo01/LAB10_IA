import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class RN(nn.Module):
    def __init__(self):
        super(RN, self).__init__()
        # We will need 2 convolutional layers, one pooling (twice) and two fully connected
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # Here we define the real arquitecture
        # Over a conv, pass a relu activation and then pooling
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # view is used to reshape the tensor x into a 2D tensor of shape 
        x = x.view(-1, 64 * 14 * 14)
        # FC with rely
        x = nn.functional.relu(self.fc1(x))
        # FC without any activation
        x = self.fc2(x)
        return x