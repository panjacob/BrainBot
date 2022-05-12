"""
Here are some Models in no particular order.
Workig so far -> OneDNetScaled
"""
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
        x = torch.stack([_.flatten()[:-17].reshape(800, 800) for _ in x.unbind()]) #create squeres
        x = x.unsqueeze_(1) #add channel dim
        #y = y.flatten()[:-17].reshape(800, 800)
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Sigmoid()
            #nn.Softmax(0) #normalize to [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = 732 # imput data size
        x = torch.stack([_.flatten()[:-abs((x.size(2) * x.size(1)) - (px*px))].reshape(px, px) for _ in x.unbind()])  # create squeres
        x = x.unsqueeze_(1)  # add channel dim
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if x.size(1) == 1: x = x.flatten() #remove channel dim
        return x


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 12.
        self.layer_1 = nn.Linear(1024, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        si = 1024  # imput data size
        x = torch.stack([_.flatten()[:-abs((x.size(2) * x.size(1)) - si)].reshape(si) for _ in x.unbind()])
        x = x.unsqueeze_(1)  # add channel dim
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


class OneDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv1d(21,42, kernel_size=(3,)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3), # downsize 3 times

            nn.Conv1d(42, 84, kernel_size=(5,)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5),  # downsize 5 times

            nn.Conv1d(84, 100, kernel_size=(5,)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5),  # downsize 5 times
        )

        self.avgpool = nn.AdaptiveAvgPool1d(100)

        self.classifier = nn.Sequential(
            nn.Linear(100 * 100 , 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )


    def forward(self, x : Tensor) -> Tensor:
        #Get proper imput size:
        sc = 30000  # imput data size
        cl = x.size(1)
        x = torch.stack([ _.flatten()[:-abs((x.size(2) * x.size(1)) - (sc * x.size(1)))].reshape(cl, sc) for _ in x.unbind()])  # create squeres
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if x.size(1) == 1: x = x.flatten()  # remove channel dim
        return x


class OneDNetScaled(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(21,42, kernel_size=(3,)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), # downsize 2 times

            nn.Conv1d(42, 84, kernel_size=(3,)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # downsize 5 times

            nn.Conv1d(84, 100, kernel_size=(3,)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5),  # downsize 5 times
        )

        self.avgpool = nn.AdaptiveAvgPool1d(10)

        self.classifier = nn.Sequential(
            nn.Linear(100 * 10 , 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )


    def forward(self, x : Tensor) -> Tensor:
        #Get proper imput size:
        sc = 200  # imput data size
        cl = x.size(1)
        if x.size(2) != sc:
            x = torch.stack([ _.flatten()[:-abs((x.size(2) * x.size(1)) - (sc * x.size(1)))].reshape(cl, sc) for _ in x.unbind()])  # create squeres
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if x.size(1) == 1: x = x.flatten()  # remove channel dim
        return x
