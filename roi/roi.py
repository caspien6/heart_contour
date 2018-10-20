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
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import math


def dice_loss(target, input):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def center_distance_loss(target_center, input):
    '''Get the average loss of one batch masks'''
    reshaped_output = input.view(-1,1, 32, 32)
    m = nn.Upsample(size=(224,224))
    
    #Original picture resize
    reshaped_output = m(reshaped_output)
    reshaped_output = reshaped_output.cpu().detach().numpy()

    y_indexes, x_indexes = np.zeros((reshaped_output.shape[0])), np.zeros((reshaped_output.shape[0]))
    
    for i in range(reshaped_output.shape[0]):
        opencv_mask = np.around(reshaped_output[i,0])*255
        opencv_mask = opencv_mask.astype(np.uint8)
        #opencv_mask = opencv_mask.long()

        rect = cv2.boundingRect(opencv_mask)
        x,y,w,h = rect
        x_indexes[i] = int(x + (w)/2)
        y_indexes[i] = int(y + (h)/2)
        
    
    '''val,x_indexes = torch.max(reshaped_output,3)
    val2,y_indexes_by_batch = torch.max(val,2)
    x_indexes_by_batch = torch.gather(x_indexes,2, y_indexes_by_batch.view(-1,1,1))
    
    y_es = ((y_indexes_by_batch.float()-target_center['y'].float().cuda())**2).float()
    x_es = ((x_indexes_by_batch.float()-target_center['x'].float().cuda())**2).float()'''
    y_es = ((y_indexes-target_center['y'])**2)
    x_es = ((x_indexes-target_center['x'])**2)
    
    return torch.mean(torch.sqrt( y_es+ x_es))


class RoiLearn:
    def __init__(self):
        torch.manual_seed(23)
        self.ae_train_losses=[]
        self.ae_valid_losses = []
        self.model_train_losses=[]
        self.model_valid_losses = []
        self.model_train_dice_losses=[]
        self.model_valid_dice_losses=[]
        self.model_train_center_losses=[]
        self.model_valid_center_losses=[]
        self.model_test_losses=[]
        self.model_test_dice_losses=[]
        self.model_test_center_losses=[]
          
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
        
    
    def ae_weights2model_feature_set(self):
        # Autoencoder W2 and b2 to the original model conv1 layer features and biases.
        # From the parameters list - index 0 is the weights
        #                          - index 1 is the biases
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
    
    def plot_ae_losses(self):
        plt.plot(range(1,len(self.ae_train_losses)+1),self.ae_train_losses, label="train")
        plt.plot(range(1,len(self.ae_valid_losses)+1),self.ae_valid_losses, label="validation")
        plt.title('Autoencoder training losses')
        plt.legend()
        plt.show()
    
    def save_ae_plots(self, filename=None):
        '''Save autoencoder plots'''
        plt.plot(range(1,len(self.ae_train_losses)+1),self.ae_train_losses, label="train")
        plt.plot(range(1,len(self.ae_valid_losses)+1),self.ae_valid_losses, label="validation")
        plt.title('Autoencoder training losses')
        plt.legend()
        plt.savefig(filename)
        plt.figure()
            
    def plot_model_losses(self):
        plt.plot(range(1,len(self.model_train_losses)+1),self.model_train_losses, label="train")
        plt.plot(range(1,len(self.model_valid_losses)+1),self.model_valid_losses, label="validation")
        plt.plot(range(1,len(self.model_test_losses)+1),self.model_test_losses, label="test")
        plt.title('Model training losses')
        plt.legend()
        plt.show()
    
    def save_model_plots(self, filename=None):
        '''Save model plots'''
        plt.plot(range(1,len(self.model_train_losses)+1),self.model_train_losses, label="train")
        plt.plot(range(1,len(self.model_valid_losses)+1),self.model_valid_losses, label="validation")
        plt.plot(range(1,len(self.model_test_losses)+1),self.model_test_losses, label="test")
        plt.title('Model training losses')
        plt.legend()
        plt.savefig(filename)
        plt.figure()
        plt.plot(range(1,len(self.model_train_dice_losses)+1),self.model_train_dice_losses, label="train")
        plt.plot(range(1,len(self.model_valid_dice_losses)+1),self.model_valid_dice_losses, label="validation")
        plt.plot(range(1,len(self.model_test_dice_losses)+1),self.model_test_dice_losses, label="test")
        plt.title('Model dice losses')
        plt.legend()
        plt.savefig(filename.rsplit('.',1)[0] + '_dice' + filename.rsplit('.',1)[1])
        plt.figure()
        plt.plot(range(1,len(self.model_train_center_losses)+1),self.model_train_center_losses, label="train")
        plt.plot(range(1,len(self.model_valid_center_losses)+1),self.model_valid_center_losses, label="validation")
        plt.plot(range(1,len(self.model_test_center_losses)+1),self.model_test_center_losses, label="test")
        plt.title('Model center losses')
        plt.legend()
        plt.savefig(filename.rsplit('.',1)[0] + '_center_' + filename.rsplit('.',1)[1])
    
    def plot_model_test_losses(self,epochs, columns_number = 5):
        step = int(math.ceil(epochs/columns_number))
        X = [x for x in range(1,epochs+1,step)]
        Y = self.model_train_losses[:epochs:step]
        Z = self.model_valid_losses[:epochs:step]
        E = self.model_test_losses[:epochs:step]
        df = pd.DataFrame(np.c_[Y,Z,E], index=X, columns=['train','valid', 'test'])
        df.plot.bar()
        plt.title('Test losses')
        plt.show()
        
    def save_model_test_losses(self, filename, epochs, columns_number = 5):
        step = int(math.ceil(epochs/columns_number))
        X = [x for x in range(1,epochs+1,step)]
        Y = self.model_train_losses[:epochs:step]
        Z = self.model_valid_losses[:epochs:step]
        E = self.model_test_losses[:epochs:step]
        
        df = pd.DataFrame(np.c_[Y,Z,E], index=X, columns=['train','valid', 'test'])
        df.plot.bar()
        plt.title('Test losses')
        plt.savefig(filename)
        plt.figure()
        
    
       
    def learn_ae(self, dataset_loader, optimizer,criterion, ep = 1, BETA = 3, RHO = 0.1, dataset_validation = None,
                weight_path = None):
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
        
        self.ae_valid_losses = []
        self.ae_train_losses = []
        rho = torch.tensor([RHO for _ in range(self.autoencoder.n_hidden)]).double().cuda()
        crit2 = nn.KLDivLoss()
        
        if (dataset_validation != None):
            modes = ['train','valid']
        else:
            modes = ['train']
        
        for epoch in range(ep):
            for mode in modes:
                epoch_loss = 0
                batch_counter = 0
                if (mode == 'train'):
                    loader = dataset_loader
                    self.autoencoder.train()
                elif (mode == 'valid'):
                    loader = dataset_validation
                    self.autoencoder.eval()
                for i_batch, sample_batched in enumerate(loader):
                    # Forward
                    encoded, decoded = self.autoencoder(sample_batched['image'])
                    # Loss
                    # first loss is the loss what the user can choose
                    first_loss = criterion(self.normalize_range(sample_batched['image']), decoded)                
                    # the second loss member is the penalty loss, this helps the higher feature learning
                    sparsity_loss = crit2( F.log_softmax(torch.mean(encoded, dim = 0)) , rho)        
                    loss = (first_loss + BETA*sparsity_loss)/dataset_loader.batch_size
                    
                    batch_counter += 1
                    epoch_loss += loss.item()
                    
                    optimizer.zero_grad()
                    
                    if (mode == 'train'):
                        loss.backward()
                        optimizer.step()
                
                if (mode == 'train'):
                    self.ae_train_losses.append(epoch_loss / float(batch_counter))
                elif (mode == 'valid'):
                    self.ae_valid_losses.append(epoch_loss / float(batch_counter))
            if epoch % 5 == 0:
                print('epoch: ', epoch,' train_loss: ', self.ae_train_losses[-1], ' valid_loss: ', self.ae_valid_losses[-1])
                if (weight_path !=None and epoch > 2 and self.ae_valid_losses[-2] > self.ae_valid_losses[-1]):
                    self.save_ae_weights(weight_path)
                

    def test_roi(self, loader_test, criterion, optimizer):        
        self.model.eval()
        
        epoch_loss = 0
        batch_counter = 0
        dc_loss = 0
        center_dist_loss = 0
        
        for i_batch, sample_batched in enumerate(loader_test):
            # Forward
            out = self.model(sample_batched['image'])
            # Loss
            loss = criterion(sample_batched['mask'], out) / loader_test.batch_size
                    
            dc_loss += dice_loss(sample_batched['mask'], out).item() / loader_test.batch_size
            center_dist_loss += center_distance_loss(sample_batched['mask_center'], out)   
            batch_counter += 1
            epoch_loss += loss.item()
                    
            optimizer.zero_grad()
            
        self.model_test_losses.append(epoch_loss / float(batch_counter))
        self.model_test_dice_losses.append(dc_loss / float(batch_counter))
        self.model_test_center_losses.append(center_dist_loss / float(batch_counter))
        print('Test results =>  test_loss: ', self.model_test_losses[-1], ' test_dice_loss: ', self.model_test_dice_losses[-1])
        print('test_center_average_loss: ', self.model_test_center_losses[-1].item(), '\n')
    
                
    def learn_roi(self, dataset_loader, optimizer,criterion, ep = 1, dataset_validation = None,
                weight_path = None, TEST_COUNT=10, dataset_test = None, earlystop_info = None, save_weight_step = 20,
                 plot_filename = 'roi_plot.png', plot_test_filename = 'roi_test_losses'):
        self.model_train_losses=[]
        self.model_valid_losses = []
        self.model_train_dice_losses=[]
        self.model_valid_dice_losses=[]
        self.model_train_center_losses=[]
        self.model_valid_center_losses=[]
        self.model_test_losses = []
        self.model_test_dice_losses = []
        self.model_test_center_losses = []
        
        earlystop_counter = 0
        if (dataset_validation != None):
            modes = ['train','valid']
        else:
            modes = ['train']
        
        for epoch in range(ep):
            for mode in modes:
                
                if (mode == 'train'):
                    loader = dataset_loader
                    self.model.train()
                elif (mode == 'valid'):
                    loader = dataset_validation
                    self.model.eval()
            
                epoch_loss = 0
                batch_counter = 0
                dc_loss = 0
                center_dist_loss = 0
                for i_batch, sample_batched in enumerate(loader):
                    # Forward
                    out = self.model(sample_batched['image'])
                    # Loss
                    loss = criterion(sample_batched['mask'], out) / loader.batch_size
                    
                    dc_loss += dice_loss(sample_batched['mask'], out).item() / loader.batch_size
                    center_dist_loss += center_distance_loss(sample_batched['mask_center'], out)    
                    batch_counter += 1
                    epoch_loss += loss.item()
                    
                    optimizer.zero_grad()
                    
                    if (mode == 'train'):
                        loss.backward()
                        optimizer.step()
                        
                if (mode == 'train'):
                    self.model_train_losses.append(epoch_loss / float(batch_counter))
                    self.model_train_dice_losses.append(dc_loss / float(batch_counter))
                    self.model_train_center_losses.append(center_dist_loss / float(batch_counter))
                elif (mode == 'valid'):
                    self.model_valid_dice_losses.append(dc_loss / float(batch_counter))
                    self.model_valid_losses.append(epoch_loss / float(batch_counter))
                    self.model_valid_center_losses.append(center_dist_loss / float(batch_counter))
                    
            if dataset_test != None and epoch % int(ep/TEST_COUNT) == 0:
                self.test_roi(dataset_test, criterion, optimizer)
            elif dataset_test != None:
                self.model_test_losses.append(self.model_test_losses[-1])
                self.model_test_dice_losses.append(self.model_test_dice_losses[-1])
                self.model_test_center_losses.append(self.model_test_center_losses[-1])
                
            if epoch % 5 == 0:
                print('epoch: ', epoch,' train_loss: ', self.model_train_losses[-1], ' valid_loss: ', self.model_valid_losses[-1])
                print('train_dice_loss: ', self.model_train_dice_losses[-1], ' valid_dice_loss: ', self.model_valid_dice_losses[-1])
                print('train_center_loss: ', self.model_train_center_losses[-1].item(), ' valid_center_loss: ', self.model_valid_center_losses[-1].item(), '\n')
            
            '''Early stopping section'''
            if (earlystop_info != None and epoch > 2 and earlystop_info['step'] > (self.model_valid_losses[-2] - self.model_valid_losses[-1]) ):
                earlystop_counter += 1   
            elif (earlystop_info != None and epoch > 2 and earlystop_info['step'] <= (self.model_valid_losses[-2] - self.model_valid_losses[-1])):
                earlystop_counter = 0
                
            if epoch % save_weight_step == 0 and epoch > 3 and weight_path != None and 0 < (self.model_valid_losses[-2] - self.model_valid_losses[-1]) :
                self.save_model_weights(weight_path)
            
            if earlystop_info != None and earlystop_info['patience_steps'] <= earlystop_counter:
                print('Early stopping!\n Epoch: ', epoch)
                self.save_model_weights(weight_path)
                self.save_model_plots(plot_filename)
                self.save_model_test_losses(plot_test_filename, epoch+1, TEST_COUNT)
                return

        self.save_model_weights(weight_path)
        if (dataset_test != None):
            self.save_model_test_losses(plot_test_filename, ep, TEST_COUNT)
        self.save_model_plots(plot_filename)
