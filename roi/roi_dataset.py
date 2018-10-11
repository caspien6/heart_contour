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

class RoiDataset(Dataset):
    """Roi dataset."""

    def __init__(self, csv_file, transform=None, target_transform=None, smpl = 0):
        """
        Args:
            csv_file (string): Path to the csv file with contour data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        'height is  (y length) width is  (x length)'
        self.contour_data = pd.read_csv(csv_file,sep=';', names=('path','slice', 'frame', 'xmin', 'ymin', 'height','width' ))
        if smpl != 0:
            self.contour_data = self.contour_data.sample(smpl)

        self.dcm_images = {}
        for path in self.contour_data['path']:
            temp_ds = dicom.dcmread(path)
            self.dcm_images[str(path)] = temp_ds.pixel_array
            
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.contour_data)
    
    def __getitem__(self, idx):
        if isinstance(idx,torch.Tensor):
            idx = idx.item()
        cont = self.contour_data.iloc[idx]
        sl = int(cont['slice'])
        fr = int(cont['frame'])
        
        mask = np.zeros((224,224))
        mask[ int(cont['ymin']): int(cont['ymin']) + int(cont['height']),int(cont['xmin']): int(cont['xmin']) + int(cont['width'])] = 1
        image = self.dcm_images[str(cont['path'])]
        
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample['image'] = self.transform(sample['image']).cuda()
        if self.target_transform:
            sample['mask'] = self.target_transform(sample['mask']).cuda()

        return sample
        