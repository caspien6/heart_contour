

class RoiBase:
    '''
    Base class for cropping the ROI of the 
    raw input image.
    '''
    def get_output_size(self):
        raise NotImplementedError()

    def get_roi(self, raw_image):
        raise NotImplementedError()
