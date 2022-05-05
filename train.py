import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from brainset import *
from model import ECGNet


class AverageMeter:
    """Computes and stores the average and current value"""

    def init(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def str(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.dict)

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


def accuracyxd(output, target, batch_size, topk=(1,), ):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():  # disables recalculation of gradients
        maxk = max(topk)
        batchsize = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res

torch.set_default_dtype(torch.float32)
brainloader, testloader = loadData()
device = torch.device("cpu")
model = ECGNet()
criterion = nn.BCELoss()
optimalizer = torch.optim.Adam(model.parameters(), lr=3e-4)
model.to(device)
inputs, labels, filenames  = next(iter(brainloader))
# average_meter = AverageMeter()
for epoch in range(100):
    print('epoch', epoch)

    accuracy = []
    loses = []
    model.train()
    accuracy = []
    loses = []
    for inputs, labels, filenames in brainloader:
        inputs = torch.autograd.Variable(inputs.to(device, non_blocking=True))
        labels = torch.autograd.Variable(labels.to(device, non_blocking=True))

        with torch.set_grad_enabled(True):
            optimalizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            #clip_grad_norm_(model.parameters(), max_norm=1)
            optimalizer.step()
            accuracy.append(accuracy_human(labels, outputs))
            loses.append(loss)

    print('accuracy', sum(accuracy) / len(accuracy))
    print('loss', (sum(loses) / len(loses)).item())

            # print('accuracy', accuracy_human(labels, outputs))

    if not epoch % 5:
        print("Testing")
        model.eval()
        accuracy = []
        loses = []
        with torch.no_grad():
            for inputs, labels, filenames in testloader:
                #labels = torch.reshape(labels, (len(labels), 1, 1))
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                accuracy.append(accuracy_human(labels, outputs))
                loses.append(loss)
                # accuracy2 = accuracyxd(outputs, labels, len(inputs))
                # print(accuracy)
            print('test accuracy', sum(accuracy) / len(accuracy))
            print('test loss', (sum(loses) / len(loses)).item())
