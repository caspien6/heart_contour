

class ContourBase:
    '''
    Base class for classes to generate contours
    of the corresponding image.
    '''
    def get_left_contour(self, roi_image):
        raise NotImplementedError()
    
    def get_right_contour(self, roi_image):
        raise NotImplementedError()