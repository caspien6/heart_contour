

RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (0, 0, 255)


class CONreader:

    def __init__(self, file_name):

        self.container = []

        con_tag = "XYCONTOUR"
        stop_tag = "POINT"
        volumerelated_tags = [
            'Field_of_view=',
            'Image_resolution=',
            'Slicethickness=',
            'Patient_weight=',
            'Patient_height',
            'Study_description='
        ]

        self.volume_data = {
            volumerelated_tags[0]: None, 
            volumerelated_tags[1]: None, 
            volumerelated_tags[2]: None,
            volumerelated_tags[3]: None,
            volumerelated_tags[4]: None,
            volumerelated_tags[5]: None
        }

        con = open(file_name, 'r')
        
        def find_volumerelated_tags(line):
            for tag in volumerelated_tags:
                if line.find(tag) != -1:
                    value = line.split(tag)[1] # the place of the tag will be an empty string, second part: value
                    self.volume_data[tag] = value
        
        def mode2colornames(mode):
            if mode == 0:
                return 'red' # check the colors out
            elif mode == 1:
                return 'green'
            elif mode == 5:
                return 'yellow'
            else:
                print('Warning: Unknown mode.')
                return 'other'

        def find_xycontour_tag():
            line = con.readline()
            find_volumerelated_tags(line)
            while line.find(con_tag) == -1 and line.find(stop_tag) == -1 and line != "":
                line = con.readline()
                find_volumerelated_tags(line)
            return line

        def identify_slice_frame_mode():
            line = con.readline()
            splitted = line.split(' ')
            return int(splitted[0]), int(splitted[1]), mode2colornames(int(splitted[2]))

        def number_of_contour_points():
            line = con.readline()
            return int(line)

        def read_contour_points(num):
            contour = []
            for _ in range(num):
                line = con.readline()
                xs, ys = line.split(' ')
                contour.append((float(xs), float(ys)))
            return contour

        line = find_xycontour_tag()
        while line.find(stop_tag) == -1 and line != "":

            slice, frame, mode = identify_slice_frame_mode()
            num = number_of_contour_points()
            contour = read_contour_points(num)
            self.container.append((slice, frame, mode, contour))
            line = find_xycontour_tag()

        con.close()
        return

    def get_contours(self):
        return self.container

    def get_hierarchical_contours(self):
        data = {}

        for item in self.container:
            slice = item[0]
            frame = item[1]
            mode = item[2]
            contour = item[3]

            # rearrange the contour
            d = {'x': [], 'y': []}
            for point in contour:
                d['x'].append(point[0])
                d['y'].append(point[1])

            if not(slice in data.keys()):
                data[slice] = {}

            if not(frame in data[slice].keys()):
                data[slice][frame] = {}

            if not(mode in data[slice][frame].keys()):
                data[slice][frame][mode] = []

            data[slice][frame][mode].append(d)

        return data
    
    def get_volume_data(self):
        # process field of view
        fw_string = self.volume_data['Field_of_view=']
        sizexsize_mm = fw_string.split('x') # variable name shows the format
        size_h = float(sizexsize_mm[0])
        size_w = float(sizexsize_mm[1].split(' mm')[0]) # I cut the _mm ending

        # process image resolution
        img_res_string = self.volume_data['Image_resolution=']
        sizexsize = img_res_string.split('x')
        res_h = float(sizexsize[0])
        res_w = float(sizexsize[1])

        # process slice thickness
        width_string = self.volume_data['Slicethickness=']
        width_mm = width_string.split(' mm')
        width = float(width_mm[0])

        # process weight
        weight_string = self.volume_data['Patient_weight=']
        weight_kg = weight_string.split(' kg')
        weight = float(weight_kg[0])

        # process height
        if 'Patient_height=' in self.volume_data.keys():
            height_string = self.volume_data['Patient_height=']
            height = height_string.split(" ")[0]
        else:
            height_string = str(self.volume_data['Study_description='])
            height = ''
            for char in height_string:
                if char.isdigit():
                    height += char
        if height == '':
            height = 178
        else:
            height = float(height)
        
        return (size_h, size_w), (res_h, res_w), width, weight, height
