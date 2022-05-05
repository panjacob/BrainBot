from torch import nn

from brainset import *
from model import ECGNet

brainloader, testloader = loadData()
device = torch.device("cpu")
model = ECGNet()
criterion = nn.CrossEntropyLoss()
optimalizer = torch.optim.Adam(model.parameters(), lr=3e-4)
model.to(device)
for epoch in range(1000):
    model.train()
    for batch in brainloader:
        inputs, labels, filenames = batch
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimalizer.step()
            optimalizer.zero_grad()


