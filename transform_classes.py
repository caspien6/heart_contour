import torch
import torch.nn as nn
import numpy as np


from sklearn.preprocessing import StandardScaler
from skimage.transform import resize


class ReScale64(object):
    """Scale down ndarrays in sample to Tensors."""

    def __call__(self, data):
        data = resize(data, (64, 64))
        return data


class ReScale32(object):
    """Scale down ndarrays in sample to Tensors."""

    def __call__(self, data):
        data = resize(data, (32, 32))
        data = np.reshape(data, (1024))
        return data
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return torch.from_numpy(data)
    
class StandardScale(object):
    """Standard scale ndarrays."""

    def __call__(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        
        # add color axis because
        # numpy image: H x W
        # torch image: C X H X W
        data = np.expand_dims(data, axis=0)
        return data
