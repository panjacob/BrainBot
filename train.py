import numpy as np
import torch
from torch import nn

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


def label_to_human_form(labels):
    result = []
    for x in labels:
        for y in x:
            result.append(int(y))
    return result


def accuracy_human(a, b):
    result = 0
    for x, y in zip(label_to_human_form(a), label_to_human_form(b)):
        result += 1 if x == y else 0
    return result / len(a)


brainloader, testloader = loadData()
device = torch.device("cpu")
model = ECGNet()
criterion = nn.BCELoss()

optimalizer = torch.optim.Adam(model.parameters(), lr=3e-4)
model.to(device)
# average_meter = AverageMeter()
for epoch in range(100):
    print('epoch', epoch)
    model.train()
    accuracy = []
    loses = []
    for inputs, labels, filenames in brainloader:
        labels = torch.reshape(labels, (len(labels), 1, 1))

        with torch.set_grad_enabled(True):
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimalizer.step()
            optimalizer.zero_grad()

            # print('accuracy', accuracy_human(labels, outputs))

    if not epoch % 5:
        model.eval()
        with torch.no_grad():
            for inputs, labels, filenames in testloader:
                labels = torch.reshape(labels, (len(labels), 1, 1))
                l1, l2 = labels
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.float())
                loses.append(loss)
                accuracy.append(accuracy_human(labels, outputs))
            # print('accuracy', sum(accuracy) / len(accuracy))

    print('loss', sum(loses) / len(loses))
    loses = []