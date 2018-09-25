import numpy as np
import png
import pydicom
from sklearn.preprocessing import normalize

from os import listdir
from os.path import isfile, join
import os
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RoiLearn:
    def __init__(self):
        torch.manual_seed(12)
        self.conv1 = nn.Conv2d(1,100, (11,11))
        self.softmax = nn.Softmax()
        self.avgpool = nn.AvgPool2d(6)
        self.flatten = Flatten()
        self.full = nn.Linear(8100,1024)
          
    def build_model(self):
        self.model = nn.Sequential(self.conv1,
                            self.avgpool,
                            self.softmax,
                            self.flatten,
                            self.full,
                            self.softmax
                            )
        self.model = self.model.double()
    
    def propagate_from_dataLoader(self,dl):
        for i_batch, sample_batched in enumerate(dl):
            print(self.model(sample_batched[0]))
        
    def propagate(self):
        return self.model(self.x)
    
    def save_image( self,npdata, outfilename ) :
        img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
        img.save( outfilename )
