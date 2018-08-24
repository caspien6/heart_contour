import dicom_reader
import con_reader
import con2img
from matplotlib.pyplot import plot, imshow, show


def test_dicom_reader():

    path = "../data/Eredetikepek/1___AW1295938826.919.1531920932/2018-07-16_1.2.840.113619.2.181.60341967613.14520.1531920932966.2/"
    dc = dicom_reader.DCMreader(path)
    print("Reading was finished!")

    imshow(dc.get_image(2, 23))
    show()

#test_dicom_reader()


def test_con_reader():

    path = "../data/Kontur/Buday_Krisztina_20180712_sa_SER1001_ACQ10.con"
    cn = con_reader.CONreader(path)
    print("Reading was finished!")

    crs = cn.get_hierarchical_contours()
    x, y = crs[7][11][0]['x'], crs[7][11][0]['y']
    plot(x, y, 'ro')
    show()

#test_con_reader()


def test_con2img():
    dcm_path = "../data/Eredetikepek/1___AW1295938826.919.1531920932/2018-07-16_1.2.840.113619.2.181.60341967613.14520.1531920932966.2/"
    con_path = "../data/Kontur/Horvath_Daniel_Konstantin_20180716_HV_SFI_SER1001_ACQ10.con"

    con2img.draw_contours2images(dcm_path, con_path)
    print("Done.")

test_con2img()