from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_wrangling.dicom_reader import DCMreader
import pydicom as dicom
from transform_classes import GetRandomPatch

class AEDataset(Dataset):
    """Autoencoder dataset."""

    def __init__(self, csv_file, transform=None, target_transform=None, sample_size = 10, random_patcher_size = 30):
        """
        Args:
            csv_file (string): Path to the csv file with contour data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #height is  (y length) width is  (x length)
        self.contour_data = pd.read_csv(csv_file,sep=';', names=('path','slice', 'frame', 'xmin', 'ymin', 'height','width' ))
        self.contour_data = self.contour_data.sample(sample_size, replace = True)
        self.smpl_size = sample_size
        self.dcm_images = []
        randomPatcher = GetRandomPatch(random_patcher_size)
        for path in self.contour_data['path']:
            temp_ds = dicom.dcmread(path)
            img = randomPatcher.__call__(temp_ds.pixel_array)
            self.dcm_images.append(img)
            
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.contour_data)
    
    def __getitem__(self, idx):
        if isinstance(idx,torch.Tensor):
            idx = idx.item()
        cont = self.contour_data.iloc[idx]        
        image = self.dcm_images[idx]
        
        sample = {'image': image}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform:
            sample['mask'] = self.target_transform(sample['mask'])

        return sample
        