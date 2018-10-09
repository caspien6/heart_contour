from data_wrangling.con_reader import CONreader
from data_wrangling.dicom_reader import DCMreader
import utils
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

    def __init__(self, contour, size, border):
        '''
        contour - class to segment the heart in the ROI
        '''
        self.get_left_contour = contour.get_left_contour
        self.get_right_contour = contour.get_right_contour
        self.size = (size, size)
        self.border = border

    def contouring_patient(self, folder):
        '''
        folder - assumed it is a folder with the same structure 
                 as the folders after data cleansing. This folder 
                 corresponds to only one patient.
        '''

        # Find .con file and the dcm files with images.
        con_file = None
        dcm_folder = None
        for root, folders, files in os.walk(folder):
            for file_name in files:
                if file_name.lower().endswith('.con'):
                    con_file = os.path.join(root, file_name)
                    dcm_folder = os.path(root, folders[0])
                    assert len(folders) == 1, 'Wrong folder structure in %s'%folder
        
        # Read the con file and the dcms
        con = CONreader(con_file)
        dcm = DCMreader(dcm_folder)

        # Create dictionary for contours
        contours = {}

        # Iterate through all the slices and frames
        hierarchical = con.get_hierarchical_contours()
        for slice in hierarchical.keys():
            if not(slice in contours.keys()):
                contours[slice] = {}
            for frame in hierarchical[slice]:
                if not(frame in contours.keys()):
                    contours[slice][frame] = {}

                # Process the current image
                image = dcm.get_image(slice, frame)
                roi_image = utils.get_roi(image, hierarchical[slice][frame], self.size, self.border)
                
                if 'red' in hierarchical[slice][frame].keys():                  # red means left
                    contour_left = self.get_left_contour(roi_image)             # get the control points
                    contours[slice][frame]['red'] = utils.spline(contour_left)  # fit a spline on the control points
                if 'yellow' in hierarchical[slice][frame].keys():
                    contour_right = self.get_right_contour(roi_image)           
                    contours[slice][frame]['yellow'] = utils.spline(contour_right)

        # return the results as CONreader objects
        p_con = ProxyCONreader(contours, [con.get_volume_data()])
        return {'manual': con, 'auto': p_con}
                



