import torch
import torch.nn as nn
import numpy as np


from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from sklearn.feature_extraction.image import extract_patches_2d


class GetRandomPatch(object):
    """Get a random 11x11 patch from an image"""

    def __call__(self, image):
        image = extract_patches_2d(image, (11, 11), 1)
        
        image = np.reshape(image, (121))
        image = image.astype(float)
        image = np.expand_dims(image , axis = 1)
        return image
    
class StandardScale2(object):
    """Standard scale ndarrays."""

    def __call__(self, data):
        scaler = StandardScaler()
        #data = data.reshape((1, -1))
        # first fit our 2 dimension data (121,1), after that change axis and delete the useless dimension
        # this magic is because the scaler.fit_transform function don't accept 1 dimensional arrays.
        data = np.reshape(scaler.fit_transform(data).swapaxes(0,1),(121))
        return data


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
