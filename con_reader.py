

RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (0, 0, 255)


class CONreader:

    def __init__(self, file_name):

        self.container = []

        con_tag = "XYCONTOUR"
        stop_tag = "POINT"

        con = open(file_name, 'r')

        def find_xycontour_tag():
            line = con.readline()
            while line.find(con_tag) == -1 and line.find(stop_tag) == -1:
                line = con.readline()
            return line

        def identify_slice_frame():
            line = con.readline()
            splitted = line.split(' ')
            return int(splitted[0]), int(splitted[1])

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

        tag = find_xycontour_tag()
        while tag.find(stop_tag) == -1:

            slice, frame = identify_slice_frame()
            num = number_of_contour_points()
            contour = read_contour_points(num)
            self.container.append((slice, frame, contour))
            tag = find_xycontour_tag()

        con.close()
        return

    def get_contours(self):
        return self.container

    def get_hierarchical_contours(self):
        data = {}

        for item in self.container:
            slice = item[0]
            frame = item[1]
            contour = item[2]

            # rearrange the contour
            d = {'x': [], 'y': []}
            for point in contour:
                d['x'].append(point[0])
                d['y'].append(point[1])

            if not(slice in data.keys()):
                data[slice] = {}

            if not(frame in data[slice].keys()):
                data[slice][frame] = []

            data[slice][frame].append(d)

        return data
