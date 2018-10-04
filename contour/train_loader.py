from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import os


class ContourTrainLoader:

    def __init__(self, folder, validation_ratio=0.2):
        '''
        Initiates a train loader for loading the training and test set.
        '''
        self.folder = folder
        self.validation_ratio = validation_ratio

        self._to_ram()
        self._split_data()
        self._torch_tensor()

    def _to_ram(self):
        '''
        Reads the images into numpy arrays.
        Reads the control points into numpy vectors.
        Images and contours are related via the indices.
        '''
        self.images = []
        self.left_contours = []
        self.right_contours = [] 
        
        # find the contours csv
        df = None
        for root, _, files in os.walk(self.folder):
            for file_name in files:
                if file_name.lower().endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, file_name), index_col=0)
                    break
        assert df is not None, 'There is no csv file!'

        # reading the images and contours
        # define function for reading the contour points
        def read_contours(name):
            num_points = len(df.columns) // 4
            left_contour = []
            right_contour = []
            for i in range(num_points):
                xl = df.loc[name]['lvx' + str(i)] 
                yl = df.loc[name]['lvy' + str(i)]
                xr = df.loc[name]['rvx' + str(i)]
                yr = df.loc[name]['rvy' + str(i)]
                left_contour.append((xl, yl))
                right_contour.append((xr, yr))

            return np.array(left_contour), np.array(right_contour)
        
        # read in the images and the corresponding contours
        for root, _, files in os.walk(self.folder):
            for file_name in files:
                if file_name.lower().endswith('.npy'):
                    img = np.load(os.path.join(root, file_name), allow_pickle=False)
                    self.images.append(img)
                    temp_left, temp_right = read_contours(file_name)
                    self.left_contours.append(temp_left)
                    self.right_contours.append(temp_right)

    def _split_data(self):
        # new datasets
        self.images_val = []
        self.images_train = []
        self.left_contours_val = []
        self.left_contours_train = []
        self.right_contours_val = []
        self.right_contours_train = []

        # shuffle the indices
        indices = [idx for idx in range(len(self.images))]
        np.random.shuffle(indices)

        num_validation = int(len(indices) * self.validation_ratio)

        # create the validation set
        for idx in range(num_validation):
            self.images_val.append(self.images[indices[idx]])
            self.left_contours_val.append(self.left_contours[indices[idx]])
            self.right_contours_val.append(self.right_contours[indices[idx]])
        
        # create the training set
        for idx in range(num_validation, len(indices)):
            self.images_train.append(self.images[indices[idx]])
            self.left_contours_train.append(self.left_contours[indices[idx]])
            self.right_contours_train.append(self.right_contours[indices[idx]])

    def _torch_tensor(self):
        '''
        Converts data into torch tensors.
        '''
        transform = lambda x: torch.from_numpy(x).unsqueeze(0)
        self.images_val = list(map(transform, self.images_val))
        self.images_train = list(map(transform, self.images_train))

        self.left_contours_val = list(map(torch.from_numpy, self.left_contours_val))
        self.left_contours_train = list(map(torch.from_numpy, self.left_contours_train))
        self.right_contours_val = list(map(torch.from_numpy, self.right_contours_val))
        self.right_contours_train = list(map(torch.from_numpy, self.right_contours_train))
        
    # Inner class for wrapping the data in a dataset
    class ContourDataset(Dataset):
        '''
        Dataset for wrapping the images and control points.
        This makes possible to use DataLoader which has several useful properties.
        '''

        def __init__(self, images, left_contours, right_contours):
            '''
            images - list with torch Tensors
            left_contours - list with torch Tensors
            right_contours - list with torch Tensors
            '''
            self.images = images
            self.left_contours = left_contours
            self.right_contours = right_contours
            assert len(images) == len(left_contours) and len(images) == len(right_contours), 'Different sizes of input vectors.'
        
        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            sample = {
                'image': self.images[idx],
                'left': self.left_contours[idx],
                'right': self.right_contours[idx]
            }
            return sample

    def get_trainloader(self, batch_size):
        '''
        Creates a data loader to iterate through the training part of the dataset.
        '''
        dataset = self.ContourDataset(self.images_train, self.left_contours_train, self.right_contours_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_validationloader(self,  batch_size):
        '''
        Creates a data loader to iterate through the test part of the dataset.
        '''
        dataset = self.ContourDataset(self.images_val, self.left_contours_val, self.right_contours_val)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
