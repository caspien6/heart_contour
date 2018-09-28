from data_wrangling import dicom_reader
from data_wrangling import con_reader
import pandas as pd
import numpy as np
import shutil
import os


class TransformRaw:

    def __init__(self, src, dst):
        '''
        Transforms and copies the data from the center database to a spearate place
        for immediate usage when training is happening.
        src - source database (should store patient data: folder:studyid -> subfolder: seriesid -> subfolder:imgs -> files:*dcm)
        dst - destination folder (structure will be dst -> file:contours.csv, folder:images)
        '''
        self.src = src
        self.dst = dst

        self._explore_cons()
        self._build_dst_structure()
    
    def _explore_cons(self):
        '''
        Looks up the .con files and count them.
        '''
        self.num_cons = 0
        for _, _, files in os.walk(self.src):
            for file_names in files:
                if file_names.lower().endswith('.con'):
                    self.num_cons += 1
        print("Source was explored. %d pieces of con files were found."%self.num_cons)
        assert self.num_cons > 0, "There are no con files!"
    
    def _build_dst_structure(self):
        '''
        Clears and rebuilds the subfolders in the destination folder.
        '''
        if os.path.exists(self.dst):
            shutil.rmtree(self.dst)  # delete the previous version if exists
        os.mkdir(self.dst)
        os.mkdir(os.path.join(self.dst, 'images'))  # create the images folder
        print("Destination folder is ready.")

    # ----------------------------------------------------
    # MAIN functionality

    def executor(self):
        '''
        Iterates over all of the files and executes the filters to process the current element.
        '''
        for root, folders, files in os.walk(self.src):
            for file_name in files:
                if file_name.lower().endswith('.con'):
                    assert len(files) == 1, "Only one con file should exist for a series id."
                    assert len(folders) == 1, "Only an imgs folder should exist."
                    dcm_folder = os.path.join(root, folders[0])
                    con_file = os.path.join(root, file_name)

                    self.dcms = dicom_reader.DCMreader(dcm_folder)
                    self.cons = con_reader.CONreader(con_file)
                    self._iterator()

        print("Execution was done.")
    
    def _iterator(self):
        '''
        Iterates over the slices and frames after a con file was selected by executor.
        After choosing the next slice and frame it executes the filter methods.
        '''
        pass
    
    def _filter_bbox(self):
        '''
        Find the bounding box around the contours.
        '''
        pass
    
    def _filter_crop_roi(self):
        '''
        Crop the image according to the bounding box.
        '''
        pass

    def _filter_rescale(self):
        '''
        Rescale image to size 100 x 100.
        '''
        pass
    
    def _filter_save(self):
        '''
        Saves the image into the destination folder. (dst/images/)
        '''
        pass

    def _filter_control_points(self):
        '''
        Find control points in the contours. (two contours: red, yellow)
        Control points: the points from which the original curve can be 
        reconstructed by a bspline with tolerably small error.
        These points will be the target of the model.
        '''
        pass
    
    def _filter_newrow(self):
        '''
        The control points for each image is stored as a row in a csv file.
        '''
        pass

