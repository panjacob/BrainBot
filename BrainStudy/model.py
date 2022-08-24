"""
Neural network models
"""
import os

import torch
import torch.nn as nn
from torch import Tensor
import copy
from datetime import datetime
import numpy as np


class OneDNetScaled(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(16, 42, kernel_size=(3,)),
            #nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.MaxPool1d(kernel_size=2),  # downsize 2 times

            nn.Conv1d(42, 84, kernel_size=(3,)),
            #nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.MaxPool1d(kernel_size=2),  # downsize 5 times

            nn.Conv1d(84, 100, kernel_size=(3,)),
            #nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.MaxPool1d(kernel_size=5),  # downsize 5 times
        )

        self.avgpool = nn.AdaptiveAvgPool1d(10)

        self.classifier = nn.Sequential(
            nn.Linear(100 * 10, 500),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),

            nn.Linear(500, 50),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),

            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        self.saved = False
        self.save_dir_path = None
        self.now_str = None

    def forward(self, x: Tensor) -> Tensor:
        # Get proper input size:
        sc = 200  # input data size
        cl = x.size(1)
        if x.size(2) != sc:
            x = torch.stack([_.flatten()[:-abs((x.size(2) * x.size(1)) - (sc * x.size(1)))].reshape(cl, sc) for _ in
                             x.unbind()])  # create squares
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if x.size(1) == 1:
            x = x.flatten()  # remove channel dim
        return x

    def saveModel(self, save_dir_path : str = '', param : str = '', create_dir = False):
        # Check if saving for the first time
        if self.saved is False:
            # Current time str - Used to differentiate saved models
            self.now_str = datetime.now().strftime("%d.%m.%Y_%H.%M")
            #Dir Name:
            dir_name = "Train_OneDNetScaled_" + self.now_str
            self.save_dir_path = save_dir_path+'/'+dir_name
            self.saved = True
            os.mkdir(self.save_dir_path)
        # Create Savable copy of model
        model_states = copy.deepcopy(self.state_dict())
        # Create New Fancy Save Name
        save_name = "Model_OneDNetScaled_" + self.now_str + '_' + param + '.pt'
        # Create Save Dir
        if create_dir:
            os.mkdir("save_dir_path")
        # Create Save path
        save_path =  self.save_dir_path+'/'+save_name
        #Save model
        torch.save(model_states, save_path)

    def loadModel(self,load_path,load_device):
        model_states = torch.load(load_path, map_location=load_device)
        self.load_state_dict(model_states)

