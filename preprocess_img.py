import data_wrangling.dicom_reader as dicom_reader
import data_wrangling.con_reader as con_reader
from data_wrangling.con2img import draw_square
from matplotlib.pyplot import plot, imshow, show,imsave
import csv
import os
import glob

def write_all_rectangle2file(rootdir):
    rectangle_file = rootdir + 'rectangle.csv'
    for subdir, dirs, files in os.walk(rootdir):
        dcm_folder = glob.glob(subdir + './*.dcm')
        if len(dcm_folder) != 0 and len(glob.glob(subdir.rsplit('\\', 1)[0] + '/*.con')) != 0:
            dcm_folder = subdir 
            con_file = glob.glob(dcm_folder.rsplit('\\', 1)[0] + '/*.con')[0]
            write_rectangle2file(dcm_folder+ '\\', con_file, rectangle_file)
        

#Write embracing rectangle info into a rectang.csv file
#dcm_folder - location where your .dcm images take place
#con_file - file which associated with the above .dcm images
#rectangle_file - the location where you want to place the rectangle info file
def write_rectangle2file(dcm_folder, con_file, rectangle_file):

    dc = dicom_reader.DCMreader(dcm_folder)
    print("Dicom files were read in!")
    cn = con_reader.CONreader(con_file)
    print("Con files were read in!")

    hierarchical = cn.get_hierarchical_contours()
        
    for slice in hierarchical.keys():
        for frame in hierarchical[slice].keys():

            img = dc.get_image(slice, frame)
            xglobalmin = 20000
            xglobalmax = 0
            yglobalmin = 20000
            yglobalmax = 0
            for mode in hierarchical[slice][frame].keys():
                # find min-max for rectangle
                
                for contour in hierarchical[slice][frame][mode]:
                    x_vec = contour['x']
                    y_vec = contour['y']

                    min_x,min_y = 20000, 20000
                    max_x,max_y = 0 , 0
                    #find local min-max
                    for idx in range(len(x_vec)):

                        if x_vec[idx] > max_x:
                            max_x = x_vec[idx]
                        if min_x > x_vec[idx]:
                            min_x = x_vec[idx]

                        if y_vec[idx] > max_y:
                            max_y = y_vec[idx]
                        if min_y > y_vec[idx]:
                            min_y = y_vec[idx]
                            
                #comparism with global min-max
                if max_x > xglobalmax:
                    xglobalmax = max_x
                if xglobalmin > min_x:
                    xglobalmin = min_x
                if max_y > yglobalmax:
                    yglobalmax = max_y
                if yglobalmin > min_y:
                    yglobalmin = min_y

            for x_coord in range(int(xglobalmax - xglobalmin)):
                for y_coord in range(int(yglobalmax - yglobalmin)):
                    draw_square(img, int(y_coord + yglobalmin), int(x_coord + xglobalmin))
            
            with open(rectangle_file, 'a',newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=';')
                #slice, frame number, x, y, height, widht
                wr.writerow([dc.get_dcm_path(slice,frame),slice, frame, xglobalmin, yglobalmin, \
                             int(yglobalmax - yglobalmin), int(xglobalmax - xglobalmin)])



#Draw embracing rectangle around contours.
#dcm_folder - location where your .dcm images take place
#con_file - file which associated with the above .dcm images
def draw_contour_rectangle2images(dcm_folder, con_file):

    dc = dicom_reader.DCMreader(dcm_folder)
    print("Dicom files were read in!")
    cn = con_reader.CONreader(con_file)
    print("Con files were read in!")

    hierarchical = cn.get_hierarchical_contours()

    for slice in hierarchical.keys():
        for frame in hierarchical[slice].keys():

            img = dc.get_image(slice, frame)
            xglobalmin = 20000
            xglobalmax = 0
            yglobalmin = 20000
            yglobalmax = 0
            for mode in hierarchical[slice][frame].keys():
                # find min-max for rectangle
                
                for contour in hierarchical[slice][frame][mode]:
                    x_vec = contour['x']
                    y_vec = contour['y']

                    min_x,min_y = 20000, 20000
                    max_x,max_y = 0 , 0
                    #find local min-max
                    for idx in range(len(x_vec)):

                        if x_vec[idx] > max_x:
                            max_x = x_vec[idx]
                        if min_x > x_vec[idx]:
                            min_x = x_vec[idx]

                        if y_vec[idx] > max_y:
                            max_y = y_vec[idx]
                        if min_y > y_vec[idx]:
                            min_y = y_vec[idx]
                            
                #comparism with global min-max
                if max_x > xglobalmax:
                    xglobalmax = max_x
                if xglobalmin > min_x:
                    xglobalmin = min_x
                if max_y > yglobalmax:
                    yglobalmax = max_y
                if yglobalmin > min_y:
                    yglobalmin = min_y

            for x_coord in range(int(xglobalmax - xglobalmin)):
                for y_coord in range(int(yglobalmax - yglobalmin)):
                    draw_square(img, int(y_coord + yglobalmin), int(x_coord + xglobalmin))
                
                
            imsave('imgs/img_' + str(slice) + '_' + str(frame) + '.png', img)
