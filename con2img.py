from data_wrangling import dicom_reader
from data_wrangling import con_reader
from numpy import copy
from matplotlib.pyplot import imsave, show


def draw_square(img, x, y, size=2):
    img[x: x + size, y: y + size] = 1


def draw_contours2images(dcm_folder, con_file):

    dc = dicom_reader.DCMreader(dcm_folder)
    print("Dicom files were read in!")
    cn = con_reader.CONreader(con_file)
    print("Con files were read in!")

    hierarchical = cn.get_hierarchical_contours()

    for slice in hierarchical.keys():
        for frame in hierarchical[slice].keys():

            #img = dc.get_image(slice, frame)

            for mode in hierarchical[slice][frame].keys():

                img = copy(dc.get_image(slice, frame)) 
                
                # draw contours on image
                for contour in hierarchical[slice][frame][mode]:

                    x_vec = contour['x']
                    y_vec = contour['y']
                    for idx in range(len(x_vec)):
                        # x, y -> y, x
                        draw_square(img, int(y_vec[idx]), int(x_vec[idx]))

                imsave('imgs/img_' + str(slice) + '_' + str(frame) + '_' + str(mode) + '.png', img)
