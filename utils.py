from skimage.util import crop
from skimage.transform import resize
from scipy.interpolate import splprep, splev
import numpy as np


def get_roi(image, contours, size, border):

    def filter_bbox():
        '''
        Find the bounding box around the contours.
        '''
        def bbox(contour):
            # contour - dictionary: keys are x, y, values are lists
            x0 = min(contour['x'])  # top left corner of the bounding box
            y0 = min(contour['y'])  # top left corner
            x1 = max(contour['x'])  # right bottom corner
            y1 = max(contour['y'])  # right bottom corner
            return (x0, y0, x1, y1)
        x0, y0, x1, y1 = None, 0, 0, 0  # the overall bbox 
        for mode in contours.keys():
            box = bbox(contours[mode][0])
            if x0 is None:
                x0, y0, x1, y1 = box
            else:
                x0 = min(x0, box[0])
                y0 = min(y0, box[1])
                x1 = max(x1, box[2])
                y1 = max(y1, box[3])
        x0 = max(x0 - border, 0)
        y0 = max(y0 - border, 0)
        x1 = min(x1 + border, image.shape[1]-1)
        y1 = min(y1 + border, image.shape[0]-1)
        return (x0, y0, x1, y1)
    
    def filter_crop_roi(image, bbox):
        '''
        Crop the image according to the bounding box.
        '''
        x0, y0, x1, y1 = bbox
        return crop(image, [(y0, image.shape[0]-y1-1), (x0, image.shape[1]-x1-1)])

    def filter_rescale(image, size):
        '''
        Rescale image to size SIZE x SIZE.
        '''
        return resize(image, size)
    
    # After put them together
    bbox = filter_bbox()
    image = filter_crop_roi(image, bbox)
    return filter_rescale(image, size)


def spline(control_points):
    '''
    Fiting a continuous closed curve on the control points.
    control_points - numpy matrix with the shape (N, 2), 
                where N is the number of control points
    '''
    pts = control_points
    tck, u = splprep(pts.T, u=None, s=0.0, per=1) 
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    
    d = {'x': [], 'y': []}
    for i in range(x_new.shape[0]):
        d['x'].append(x_new[i])
        d['y'].append(y_new[i])
    return d


def draw_con2img(image, control_points):
    '''
    Drawing the control points on the given image.
    ''' 
    def draw_square(img, x, y, size=2):
        img[x: x + size, y: y + size] = 1
              
    contour = spline(control_points)
    x_vec = contour['x']
    y_vec = contour['y']
    for idx in range(len(x_vec)):
        # x, y -> y, x
        draw_square(image, int(y_vec[idx]), int(x_vec[idx]))
