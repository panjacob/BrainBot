import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from brainset import *
from model import *

single_batch_test = False

def label_to_human_form(labels):
    result = []
    for x in labels:
        #for y in x:
        result.append(int(x))
    return result


def accuracy_human(a, b):
    result = 0
    for x, y in zip(label_to_human_form(a), label_to_human_form(b)):
        result += 1 if x == y else 0
    return result / len(a)



torch.set_default_dtype(torch.float32)
brainloader, testloader = loadData()
device = torch.device("cuda")
model = AlexNet()
criterion = nn.BCELoss() #binary cross entropy
optimalizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)

if single_batch_test is False:
    # Preform Single Batch Test
    brainloader = [next(iter(brainloader))]
    print("Single Batch Test Chosen")

for epoch in range(1000):
    #if not epoch % 50:
    print('epoch', epoch)

    train_accuracy = []
    train_loses = []
    model.train()
    for inputs, labels, filenames in brainloader:

        inputs = torch.autograd.Variable(inputs.to(device, non_blocking=True))
        labels = torch.autograd.Variable(labels.to(device, non_blocking=True))

        with torch.set_grad_enabled(True):
            optimalizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #loss = 100 - 2*abs(loss - 50)
            loss.backward()
            #clip_grad_norm_(model.parameters(), max_norm=1)
            optimalizer.step()
            train_accuracy.append(accuracy_human(labels, outputs))
            train_loses.append(loss)

    #if not epoch % 50:
    print('accuracy', sum(train_accuracy) / len(train_accuracy))
    print('loss', (sum(train_loses) / len(train_loses)).item())
    model.eval()

    if not epoch % 10:
        print("Testing")
        accuracy = []
        loses = []
        model.eval()
        with torch.no_grad():
            for inputs, labels, filenames in testloader:
                inputs = torch.autograd.Variable(inputs.to(device, non_blocking=True))
                labels = torch.autograd.Variable(labels.to(device, non_blocking=True))
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                accuracy.append(accuracy_human(labels, outputs))
                loses.append(loss)
            print('test accuracy', sum(accuracy) / len(accuracy))
            print('test loss', (sum(loses) / len(loses)).item())
