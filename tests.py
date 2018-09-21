import dicom_reader
import con_reader
import con2img
import volume as vol
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

def test_con_volumedata():
    base_path = "../../data/SA_all_2/19307824AMR806/1301/"
    con_path = base_path + "contour.con"
    cn = con_reader.CONreader(con_path)
    fw, reso, width = cn.get_volume_data()
    assert fw[0] == 329.999996 and fw[1] == 329.999996, 'Wrong field view!'
    assert reso[0] == 224 and reso[1] == 224, 'Wrong image resolution!'
    assert width == 8.0, 'Wrong sice width!'
    print('OK: test_con_volumedata')

#test_con_volumedata()

def test_con2img():
    #base_path = "../../data/SA_all_2/19194862AMR806/1401/"
    base_path = "../../data/SA_all_2/19307824AMR806/1301/"
    dcm_path = base_path + "imgs/"
    con_path = base_path + "contour.con"

    con2img.draw_contours2images(dcm_path, con_path)
    print("OK: test_con2img")

#test_con2img()

def test_volume():
    path = "../../data/volume_test/Takacs_Akos_20180131_038318067_STD19194862AMR806_SER1401_ACQ14.con"
    v = vol.Volume(path)
    v.calculate_volumes('pixel')
    print(v.lved)
    print(v.lved_i)
    print(v.lves)
    print(v.lves_i)
    print(v.lvsv)
    print(v.lvsv_i)

    print(v.rved)
    print(v.rved_i)
    print(v.rves)
    print(v.rves_i)
    print(v.rvsv)
    print(v.rvsv_i)


test_volume()
