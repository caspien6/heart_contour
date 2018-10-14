import numpy as np
import png
import pydicom
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import os
import torch
import torch.nn as nn
from autoencoder import Autoencoder
from roinn import RoiNN


class RoiLearn:
    def __init__(self):
        torch.manual_seed(23)
        '''self.conv1 = nn.Conv2d(1,100, (11,11))
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(6)
        self.flatten = Flatten()
        self.full = nn.Linear(8100,1024)'''
        #self.encoder = nn.Linear(121,100)
        #self.decoder = nn.Linear(100,121)
          
    # Autoencoder architecture
    def build_ae(self):
        self.autoencoder = Autoencoder(121,100).to('cuda')
        self.autoencoder = self.autoencoder.double()
        
    def save_ae_weights(self,path):
        torch.save(self.autoencoder.state_dict(), path)

    def load_ae_weights(self,path):
        self.build_ae()
        self.autoencoder.load_state_dict(torch.load(path))

    def build_model(self):
        self.model = RoiNN().to('cuda')
        self.model = self.model.double()
        
    def save_model_weights(self,path):
        torch.save(self.model.state_dict(), path)

    def load_model_weights(self,path):
        self.build_model()
        self.model.load_state_dict(torch.load(path))
        
    # Autoencoder W2 and b2 to the original model conv1 layer features and biases.
    # From the parameters list - index 0 is the weights
    #                          - index 1 is the biases
    def ae_weights2model_feature_set(self):
        
        w2 = self.autoencoder.encoder.weight.cpu().detach().numpy()
        b2 = self.autoencoder.encoder.bias.cpu().detach().numpy()
        
        # weights shape here (100,121)
        w2 = np.expand_dims(w2.reshape((100,11,11)), axis = 1)
        # weights shape (100,1,11,11)


        self.model.conv1.weight = torch.nn.Parameter(torch.from_numpy(w2))
        self.model.conv1.bias = torch.nn.Parameter(torch.from_numpy(b2))
        self.model.conv1 = self.model.conv1.cuda()
        self.model.conv1.weight.requires_grad=False
        self.model.conv1.bias.requires_grad=False
        
    
    def normalize_range(self, vector):
        min_v = torch.min(vector)        
        range_v = torch.max(vector) - min_v
        
        if range_v > 0:
            normalised = (vector - min_v) / range_v
        else:
            normalised = torch.zeros(vector.size())
        return normalised

    ''' Learn the autoencoder features for the original convolution weights.
        Params:
            dataset_loader - the prepared dataset inside a configured pytorch dataloader
            optimizer - for the backpropagation
            criterion - method for the half part of the loss function
            ep - epochs
            lr - learning rate
            BETA - weightening the sparsity part of the loss function
            RHO - the sample distribution for comparing the average activation (sparse part of the loss function too.)
    '''     
    def learn_ae(self, dataset_loader, optimizer,criterion, ep = 1, lr = 0.01, BETA = 3, RHO = 0.1):
        
        rho = torch.tensor([RHO for _ in range(self.autoencoder.n_hidden)]).double().cuda()
        crit2 = nn.KLDivLoss(size_average=False)
        for epoch in range(ep):
            for i_batch, sample_batched in enumerate(dataset_loader):
    
                # Forward
                encoded, decoded = self.autoencoder(sample_batched['image'])
                # Loss
                # first loss is the loss what the user can choose
                first_loss = criterion(self.normalize_range(sample_batched['image']), decoded)                
                # the second loss member is the penalty loss, this helps the higher feature learning
                sparsity_loss = crit2( F.log_softmax(torch.mean(encoded, dim = 0)) , rho)        
                loss = first_loss + BETA*sparsity_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch: ', epoch,' loss: ', loss.item())
                 

    def learn_roi(self, dataset_loader, optimizer,criterion, ep = 1):
        for epoch in range(ep):
            for i_batch, sample_batched in enumerate(dataset_loader):
                # Forward
                out = self.model(sample_batched['image'])
                # Loss
                loss = criterion(sample_batched['mask'], out)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch: ', epoch,' loss: ', loss.item())
