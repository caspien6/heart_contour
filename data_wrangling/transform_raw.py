from skimage.io import imsave
from data_wrangling import dicom_reader
from data_wrangling import con_reader
import pandas as pd
import numpy as np
import shutil
import utils
import csv
import os


BORDER = 15
SIZE = 110
CONTROL_NUM = 8

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
        self.csv = None
        self.image_id = 0
        self.processed_cons = 0
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

                    self.processed_cons += 1
        self._release()
        print("Execution was done.")
    
    def _iterator(self):
        '''
        Iterates over the slices and frames after a con file was selected by executor.
        After choosing the next slice and frame it executes the filter methods.
        '''
        contours_all_slice = self.cons.get_hierarchical_contours()
        for slice in contours_all_slice.keys():
            for frame in contours_all_slice[slice].keys():
                self.image = self.dcms.get_image(slice, frame)    # the corresponding image
                self.image_id += 1
                self.contours = contours_all_slice[slice][frame]  # the red, yellow, (green) contours on the correponding image
                
                # do the processing on the corresponding image and the contours 
                self.image = utils.get_roi(self.image, self.contours, (SIZE, SIZE), BORDER) 
                self._filter_save()
                self._filter_control_points()
                self._filter_newrow()

        print("Progress: [%d%%]. \r"%(self.processed_cons/self.num_cons * 100.0), end='')
    
    def _filter_save(self):
        '''
        Saves the image into the destination folder. (dst/images/)
        '''
        self.img_name = "img_" + str(self.image_id) + ".npy"
        file_path = os.path.join(os.path.join(self.dst, "images"), "img_" + str(self.image_id) + ".npy")
        np.save(file_path, self.image, allow_pickle=False)
        #imsave(file_path, (self.image - np.min(self.image))/(np.max(self.image) - np.min(self.image)))

    def _filter_control_points(self):
        '''
        Find control points in the contours. (two contours: red, yellow)
        Control points: the points from which the original curve can be 
        reconstructed by a bspline with tolerably small error.
        These points will be the target of the model.
        '''
        def control_points(side):
            controls = []
            
            if side in self.contours.keys():
                step_size = len(self.contours[side][0]['x'])/CONTROL_NUM
                for idx in range(CONTROL_NUM):
                    control_idx = round(idx * step_size)
                    x = self.contours[side][0]['x'][control_idx]
                    y = self.contours[side][0]['y'][control_idx]
                    controls.append((x, y))
            else:
                controls = [(np.nan, np.nan)] * CONTROL_NUM
            return controls
        
        self.left_contour = control_points('red')
        self.right_contour = control_points('yellow')
    
    def _filter_newrow(self):
        '''
        The control points for each image is stored as a row in a csv file.
        '''
        if self.csv is None:
            self.f = open(os.path.join(self.dst, "contours.csv"), 'a', 1, newline='')
            self.csv = csv.writer(self.f, delimiter=',')
            header = ['index']
            for preterm in [('lvx', 'lvy'), ('rvx', 'rvy')]:
                for i in range(CONTROL_NUM):
                    header.append(preterm[0] + str(i))  # x value
                    header.append(preterm[1] + str(i))  # y value
            self.csv.writerow(header)
        row = [self.img_name] 
        for contour in [self.left_contour, self.right_contour]:
            for point in contour:
                row.append(point[0])
                row.append(point[1])
        self.csv.writerow(row)
    
    def _release(self):
        self.f.close()
      