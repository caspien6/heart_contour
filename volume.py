from data_wrangling.con_reader import CONreader
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev
import numpy as np
import math

class Volume:

    def __init__(self, con_file):
        '''
        This class is responsible for calculating metrics like:
        - ED, ED index
        - ES, ES index
        - SV (Stroke volume)
        - SV index
        - EF
        They are calculated for both the left and right side.
        '''
        self.con_file = con_file
        self.cr = CONreader(self.con_file)
        self.fview, self.img_res, self.slice_width, self.weight, self.height = self.cr.get_volume_data()
        self.contours = self.cr.get_hierarchical_contours()
    
    # ---------------------------------
    # calculate area with pixels

    def _curve2npmtx(self, curve):
        '''
        curve: dictionary with x and y keys,
               each key leads to a list with (x, y) tuples
        '''
        assert len(curve) == 1, 'Too much curves.'
        curve_ = curve[0]
        mtx = np.zeros((len(curve_['x']), 2), dtype=int)
        mtx[:, 0] = np.array(curve_['x'], dtype=int)
        mtx[:, 1] = np.array(curve_['y'], dtype=int)
        return mtx
    
    def _close_curve(self, curve):
        '''
        curve: a numpy matrix
        '''
        closed = []
        x0, y0 = curve[-1, 0], curve[-1, 1]
        idx = 0
        while idx < curve.shape[0]:
            x1, y1 = curve[idx, 0], curve[idx, 1]
            x_, y_ = x0, y0
            closed.append((x0, y0))
            if abs(x1 - x0) > 1:
                x_ = x0 + int((x1 - x0)/abs(x1 - x0))
            if abs(y1 - y0) > 1:
                y_ = y0 + int((y1 - y0)/abs(y1 - y0))
            if (abs(x1 - x0) > 1) or (abs(y1 - y0) > 1):
                x0, y0 = x_, y_
            else:
                x0, y0 = x1, y1
                idx += 1

        return np.array(closed)

    def _bbox(self, curve):
        '''
        curve: a numpy matrix with shape (N, 2), points are in x, y format
               elements are integers
        '''
        # find the corners of the bbox
        mins = np.min(curve, axis=0)
        maxs = np.max(curve, axis=0)
        x_min, y_min = mins[0], mins[1]
        x_max, y_max = maxs[0], maxs[1]

        # map to a plane with additional pixels at the border
        height = y_max - y_min + 4 # +4 pixels at the borders to help BFS
        width = x_max - x_min + 4
        plane = np.zeros((height, width), dtype=np.int32)
        for idx in range(curve.shape[0]):
            x = curve[idx, 0] - x_min + 2
            y = curve[idx, 1] - y_min + 2
            plane[y, x] = 1
        return plane
    
    def _bfs(self, plane):
        height = plane.shape[0]
        width = plane.shape[1]

        c_point = (0, 0)   # current point
        plane[c_point] = 1 # mark current point as discovered
        queue = [c_point]  # list for storing the points
        while len(queue) > 0:
            # detach next point
            c_point = queue[0]
            del queue[0]

            # discover valid neighbors
            for de in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                y = c_point[0] + de[0]
                x = c_point[1] + de[1]
                    
                is_on_plane = (x >= 0 and x < width)
                is_on_plane = (is_on_plane and y >= 0 and y < height)
                
                if is_on_plane:
                    is_unknown = (plane[y, x] == 0)
                    # if the point is valid then append it to the list
                    if is_on_plane and is_unknown:
                        # mark current point as discovered and put into FIFO
                        plane[(y, x)] = 1
                        queue.append((y, x))
        return plane

    def _calculate_area_pixel(self, curve):
        curve_mtx = self._curve2npmtx(curve)
        curve_closed = self._close_curve(curve_mtx)
        plane1 = self._bbox(curve_closed)
        plane = self._bfs(np.copy(plane1))
        h, w = plane.shape
        complementer = np.sum(plane)
        area_in_pixels = h * w - complementer + np.sum(plane1)
        if (h * w - complementer) == 0:
            plt.imshow(plane1)
            plt.show()
            plt.imshow(plane)
            plt.show()
        return float(area_in_pixels) * self.fview[0] / self.img_res[0] * self.fview[1] / self.img_res[1]
    
    # ---------------------------------
    # calculate area with triangulars

    def _curve2npmtx_float(self, curve):
        '''
        curve: dictionary with x and y keys,
               each key leads to a list with (x, y) tuples
        '''
        assert len(curve) == 1, 'Too much curves.'
        curve_ = curve[0]
        mtx = np.zeros((len(curve_['x']), 2), dtype=float)
        mtx[:, 0] = np.array(curve_['x'], dtype=float)
        mtx[:, 1] = np.array(curve_['y'], dtype=float)
        return mtx

    def __calculate_area_triangular(self, curve):
        curve_mtx = self._curve2npmtx_float(curve)
        ratio = self.fview[0] / self.img_res[0] * self.fview[1] / self.img_res[1]
        
        # calculate center of mass
        crm = np.sum(curve_mtx, axis=0) / curve_mtx.shape[0]

        # vector between crm and a point of the curve
        r = curve_mtx - crm

        # side vector
        curve_mtx_shifted = np.ones_like(curve_mtx)
        curve_mtx_shifted[0] = curve_mtx[-1]
        curve_mtx_shifted[1:] = curve_mtx[0:-1]
        dr = curve_mtx - curve_mtx_shifted

        # vector product
        rxdr = np.cross(r, dr)

        # sum up the pieces of triangulars
        area = np.abs(0.5 * np.sum(rxdr))

        return area * ratio
    
    def _caclulate_slice_volume(self, area_u, area_b):
        dV = self.slice_width * (area_u + math.sqrt(area_u * area_b) + area_b) / 3.0
        return dV
    
    def _grouping(self, calculate_area):
        contour_areas = {}
        slices = self.contours.keys()
        for slice in slices:
            contour_areas[slice] = {}
            for mode, side in zip(['red', 'yellow'], ['left', 'right']): # red: left, yellow: right
                contour_areas[slice][side] = {}
                areas = []
                frames = []
                for frame in self.contours[slice].keys():
                    if mode in self.contours[slice][frame].keys():
                        curve = self.contours[slice][frame][mode]
                        frames.append(frame)
                        areas.append(calculate_area(curve))

                if len(areas) > 1:
                    contour_areas[slice][side]['diastole'] = max(areas)
                    contour_areas[slice][side]['systole'] = min(areas)
                elif len(areas) == 1:
                    ds = np.array([frames[0] - 0, frames[0] - 23, frames[0] - 8])
                    idx = np.argmin(np.abs(ds))
                    if idx in [0, 1]:
                        contour_areas[slice][side]['diastole'] = areas[0]
                        contour_areas[slice][side]['systole'] = None
                    else:
                        contour_areas[slice][side]['diastole'] = None
                        contour_areas[slice][side]['systole'] = areas[0]
                else:
                    contour_areas[slice][side]['diastole'] = None
                    contour_areas[slice][side]['systole'] = None
        return contour_areas

    def calculate_volumes(self):
        areas_left = self._grouping(self._calculate_area_pixel)
        areas_right = self._grouping(self.__calculate_area_triangular)

        def volume(side, state):
            if side == 'left':
                areas = areas_left
            elif side == 'right':
                areas = areas_right
            else:
                raise AttributeError("Unkown side: %s"%side)

            slices = list(areas.keys())
            V = 0
            for idx in range(len(slices)-1):
                A1 = areas[slices[idx]][side][state]
                A2 = areas[slices[idx+1]][side][state]
                if (A1 is not None) and (A2 is not None):
                    V += self._caclulate_slice_volume(A1, A2)
            return V / 1000.0 # mm^3 -> ml conversion

        self.lved = volume('left', 'diastole')  # left ED
        self.lves = volume('left', 'systole')   # left ES
        self.rved = volume('right', 'diastole')
        self.rves = volume('right', 'systole')
        
        bsa = math.sqrt(self.height * self.weight/3600) # Mosteller BSA

        # other metrics: left
        self.lved_i = self.lved / bsa      # left ED-index
        self.lves_i = self.lves / bsa      # left ES-index
        self.lvsv = self.lved - self.lves  # left Stroke-volume
        self.lvsv_i = self.lvsv / bsa      # left SV-index

        # other metrics: right
        self.rved_i = self.rved / bsa      # right ED-index
        self.rves_i = self.rves / bsa      # right ES-index
        self.rvsv = self.rved - self.rves  # right Stroke-volume
        self.rvsv_i = self.rvsv / bsa      # right SV-index
