from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dicom_reader import DCMreader

class RoiDataset(Dataset):
    """Roi dataset."""

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with contour data.
            root_dir (string): Directory for .dcm images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        'height is  (y length) width is  (x length)'
        self.contour_data = pd.read_csv(csv_file,sep=';', names=('slice', 'frame', 'xmin', 'ymin', 'height','width' ))
        self.dcm_images = DCMreader(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.contour_data)
    
    def __getitem__(self, idx):
        cont = self.contour_data.iloc[idx]
        sl = int(cont['slice'])
        fr = int(cont['frame'])
        
        mask = np.zeros((224,224))
        mask[ int(cont['ymin']): int(cont['ymin']) + int(cont['height']),int(cont['xmin']): int(cont['xmin']) + int(cont['width'])] = 1
        image = self.dcm_images.get_image(sl,fr)
        
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform:
            sample['mask'] = self.target_transform(sample['mask'])

        return sample
        
