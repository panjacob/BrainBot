from torch.nn.modules.activation import Sigmoid
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class ECGNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.channels = 21
        self.hidden_size = 32
        self.kernel = 5
        self.dropout_p1 = 0.30
        self.dropout_p2 = 0.25

        self.c1 = nn.Sequential(
            nn.Conv1d(self.channels, self.hidden_size, kernel_size=(self.kernel,)),
            nn.Dropout(self.dropout_p1),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=(self.kernel,)),
            nn.Dropout(self.dropout_p2),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(self.hidden_size, 1, kernel_size=(self.kernel,)),
            nn.Dropout(self.dropout_p2),
            nn.ReLU()
        )
        '''
        self.fc1 = nn.Sequential(
            nn.Linear(16 * self.kernel, 40),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
# 29988
        '''
        self.out = nn.Sequential(
            nn.Linear(29988 , 1),
            nn.Sigmoid()
        )

    def forward(self, x : Tensor) -> Tensor:
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        #x = self.fc1(x)
        #x = self.fc2(x)
        x = self.out(x)
        return x.flatten()


class FunnyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(1, 10 , kernel_size=35, stride=4, padding=0)
        #self.conv2 = torch.nn.Conv2d(10, 40, kernel_size=11, stride=4, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=4)
        self.fc1 = torch.nn.Linear(15360, 128)
        self.fc2 = torch.nn.Linear(128,64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16,1)


    def forward(self, x : Tensor) -> Tensor:
        x = x.unsqueeze_(1)  #add channel dim
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x