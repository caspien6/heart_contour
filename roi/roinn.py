import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RoiNN(nn.Module):
    def __init__(self):
        super(RoiNN, self).__init__()
        self.conv1 = nn.Conv2d(1,100, (11,11))
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(6)
        self.flatten = Flatten()
        self.full = nn.Linear(8100,1024)

    def forward(self, x):
        convolved = F.softmax(self.avgpool(self.conv1(x)))
        out = F.softmax(self.full(self.flatten(convolved)))
        return out
