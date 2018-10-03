from torch.utils.data import TensorDataset, DataLoader

class ContourTrainLoader:

    def __init__(self, folder, validation_ratio=0.2):
        '''
        Initiates a train loader for loading the training and test set.
        '''
        self.folder = folder
        self.validation_ratio = validation_ratio

    def _to_ram(self):
        '''
        Reads the images into numpy arrays.
        Reads the control points into numpy vectors.
        '''
        pass

    def _split_data(self):
        pass

    def _torch_tensor(self):
        '''
        Converts the data into torch tensors.
        '''
        pass

    def get_trainloader(self, batch_size):
        '''
        Creates a data loader to iterate through the training part of the dataset.
        '''
        pass

    def get_testloader(self,  batch_size):
        '''
        Creates a data loader to iterate through the test part of the dataset.
        '''
        pass