from data_wrangling.con_reader import CONreader
from data_wrangling.dicom_reader import DCMreader
import os

class ProxyCONreader:
    '''
    This object stores the results of the automatic segmentation.
    It makes possible to use the results in other components like 
    the volume calculator.
    '''
    def __init__(self, contours, volume_data):
        self.contours = contours
        self.size = volume_data[0]
        self.resolution = volume_data[1]
        self.width = volume_data[2]
        self.weight = volume_data[3]
        self.height = volume_data[4]
    
    def get_hierarchical_contours(self):
        return self.contours
    
    def get_volume_data(self):
        return self.size, self.resolution, self.width, self.weight, self.height


class Executor:

    def __init__(self, roi, contour):
        '''
        roi - class to find the ROI
        contour - class to segment the heart in the ROI
        '''
        self.get_roi = roi.get_roi
        self.get_left_contour = contour.get_left_contour
        self.get_right_contour = contour.get_right_contour

    def contouring_patient(self, folder):
        pass
