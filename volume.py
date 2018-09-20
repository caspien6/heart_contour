from con_reader import CONreader
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
        self.fview, self.img_res, self.slice_width = self.__process_con()
    
    def __process_con(self):
        cr = CONreader(self.con_file)
        return cr.get_volume_data()

    def __calculate_area(self, curve):
        pass
    
    def __caclulate_slice_volume(self, area_u, area_b):
        dV = self.slice_width * (area_u + math.sqrt(area_u * area_b) + area_b)
        return dV

