from torch.nn.modules.activation import Sigmoid
import torch
import torch.nn as nn

class ECGNet(nn.Module):
  def __init__():
    self.channels = 21
    self.hidden_size = 32
    self.kernel = 5
    self.dropout_p1 = 0.30
    self.dropout_p2 = 0.25

    self.c1 = nn.Sequential(
        nn.Conv1d(self.channels, self.hidden_size, kernel_size=(self.kernel,)),
        nn.Dropout(nn.dropout_p1),
        nn.ReLU()
    )

    self.c2 = nn.Sequential(
        nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=(self.kernel,)),
        nn.Dropout(nn.dropout_p2),
        nn.ReLU()
    )

    self.c3 = nn.Sequential(
        nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=(self.kernel,)),
        nn.Dropout(nn.dropout_p2),
        nn.ReLU()
    )

    self.out = nn.Sequential(
        nn.Linear(self.hidden_size, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = self.c1(x)
    x = self.c2(x)
    x = self.c3(x)
    x = self.out(x)
    return x
