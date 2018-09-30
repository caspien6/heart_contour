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
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(6)
        self.flatten = Flatten()
        self.full = nn.Linear(8100,1024)
        self.encoder = nn.Linear(121,100)
        self.decoder = nn.Linear(100,121)
          
    # Autoencoder architecture
    def build_ae(self):
        self.autoencoder = nn.Sequential(self.encoder,
                                   self.sigmoid,
                                   self.decoder,
                                   self.sigmoid
                                  )
        self.autoencoder = self.autoencoder.double()
        
    # Autoencoder W2 and b2 to the original model conv1 layer features and biases.
    # From the parameters list - index 0 is the weights
    #                          - index 1 is the biases
    def ae_weights2model_feature_set(self):
        
        w2 = list(self.encoder.parameters())
        
        b2 = w2[1].detach().numpy()
        # weights shape here (100,121)
        w2 = np.expand_dims(w2[0].detach().numpy().reshape((100,11,11)), axis = 1)
        # weights shape (100,1,11,11)
        
        conv1_features = list(self.conv1.parameters())
        conv1_features[0] = torch.nn.Parameter(torch.from_numpy(w2))
        conv1_features[1] = torch.nn.Parameter(torch.from_numpy(b2))
        conv1_features[0].requires_grad=False
        conv1_features[1].requires_grad=False
        
    def learn_ae(self, dataset_loader,criterion, optimizer, ep = 1, lr = 0.01):
        for epoch in range(ep):
            for i_batch, sample_batched in enumerate(dataset_loader):
                #print(i_batch, sample_batched['image'].size(),sample_batched['mask'].size())
                #print(sample_batched['image'].shape)
                # Forward Propagation
                y_pred = self.autoencoder(sample_batched['image'])
                # Compute and print loss
                loss = criterion(y_pred, sample_batched['image'])
                print('epoch: ', epoch,' loss: ', loss.item())
                # Zero the gradients
                optimizer.zero_grad()

                # perform a backward pass (backpropagation)
                loss.backward()

                # Update the parameters
                optimizer.step()
    
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
