import torch
from PIL import Image
import pydicom as dicom
from sklearn.preprocessing import StandardScaler
import numpy as np
import torchvision.datasets as ds


class Preprocessor:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
    
    def preprocess_img(self,img):
        img = img.resize((64,64),Image.ANTIALIAS)
        img = img.convert('LA')
        data = np.asarray( img, dtype="float" )[:,:,:1]
        data = np.swapaxes(data, 0, 2)
        data[0] = self.scaler.fit_transform(data[0])
        return data
    
    def preprocess_dcm(self,img):
        img = Image.fromarray(img)
        img = img.convert('L')
        img = img.resize((64,64),Image.ANTIALIAS)
        img = img.convert('LA')
        data = np.asarray( img, dtype="float" )[:,:,:1]
        #channel - height swap
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 1, 2)
        data[0] = self.scaler.fit_transform(data[0])
        return data
    
    def load_image(self, infilename ):
        img = Image.open( infilename )
        img.load()
        return img
    
    def load_dcm(self,infilename):
        temp_ds = dicom.dcmread(infilename).pixel_array
        return temp_ds
    
    def create_datasetLoaderPNG(self, folder):
        dataset = ds.DatasetFolder(folder, self.load_image, ['.png'], self.preprocess_img)
        self.data = torch.utils.data.DataLoader(dataset)
        return self.data
    
    def create_datasetLoaderDCM(self, folder):
        dataset = ds.DatasetFolder(folder, self.load_dcm, ['.dcm'], self.preprocess_dcm)
        self.data = torch.utils.data.DataLoader(dataset)
        return self.data
    
    def get_dataLoader(self):
        return self.data
    
    def show_some_loadedData(self):
        if (self.data == None):
            return
        for i_batch, sample_batched in enumerate(self.data):
            print(i_batch, sample_batched[0].size(),sample_batched[1].size())
            print(sample_batched[0])
            break

