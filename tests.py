from data_wrangling import dicom_reader
from data_wrangling import con_reader
from data_wrangling import transform_raw as tr
from data_wrangling import con2img
from contour.train_loader import ContourTrainLoader
from metrics import volume as vol
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
    fw, reso, width, weight, height = cn.get_volume_data()
    assert fw[0] == 329.999996 and fw[1] == 329.999996, 'Wrong field view!'
    assert reso[0] == 224 and reso[1] == 224, 'Wrong image resolution!'
    assert width == 8.0, 'Wrong sice width!'
    print('OK: test_con_volumedata')

#test_con_volumedata()

def test_con2img():
    #base_path = "../../data/sa_all_2/19194862AMR806/1401/"
    #base_path = "../../data/sa_all_2/19307824AMR806/1301/"
    #base_path = "../../data/sa_all_1/17827245AMR809/1301/"
    #base_path = "../../data/sa_all_2/19049093AMR806/2301/"
    base_path = "../../data/sa_all_1/17128355AMR812/1001/"
    dcm_path = base_path + "imgs/"
    con_path = base_path + "contour.con"

    con2img.draw_contours2images(dcm_path, con_path)
    print("OK: test_con2img")

#test_con2img()

def test_volume():
    path1 = "../../data/volume_test/Takacs_Akos_20180131_038318067_STD19194862AMR806_SER1401_ACQ14.con"
    path2 = "../../data/SA_all/17352962AMR801/901/contour.con"
    path4 = "../../data/SA_all/17577538AMR801/901/contour.con"
    path5 = "../../data/SA_all/17651351AMR806/1301/contour.con"
    path6 = "../../data/sa_all_1/17827245AMR809/1301/contour.con"
    
    path7 = "../../data/sa_all_3/17660532AMR708/801/contour.con" # bon marcell
    path8 = "../../data/sa_all_2/19046832AMR806/801/contour.con"
    path9 = "../../data/sa_all_2/19049093AMR806/2301/contour.con"
    path10 = "../../data/sa_all_2/19046863AMR806/801/contour.con"
    path11 = "../../data/sa_all_2/19046938AMR806/801/contour.con"
    path12 = "../../data/sa_all_2/19046900AMR806/1001/contour.con"
    path13 = "../../data/sa_all_2/19050064AMR806/901/contour.con"
    path14 = "../../data/sa_all_2/19046968AMR806/801/contour.con"
    path15 = "../../data/sa_all_2/19046995AMR806/801/contour.con"

    v = vol.Volume(path15)
    v.calculate_volumes()
    print('---------------------------')
    print('LEFT VEN.')
    print('LVED: %.3f'%v.lved)
    print('LVED-idx: %.3f'%v.lved_i)
    print('LVES: %.3f'%v.lves)
    print('LVES-idx: %.3f'%v.lves_i)
    print('LVSV: %.3f'%v.lvsv)
    print('LVSV-idx: %.3f'%v.lvsv_i)
    
    print('---------------------------')
    print('RIGHT VEN.')
    print('RVED: %.3f'%v.rved)
    print('RVED-idx: %.3f'%v.rved_i)
    print('RVES: %.3f'%v.rves)
    print('RVES-idx: %.3f'%v.rves_i)
    print('RVSV: %.3f'%v.rvsv)
    print('RVSV-idx: %.3f'%v.rvsv_i)

#test_volume()

def test_transform_raw():

    src = 'src'
    dst = 'dst'

    transform = tr.TransformRaw(src, dst)
    transform.executor()

#test_transform_raw()

def test_train_loader():

    folder = 'dst'

    train_loader = ContourTrainLoader(folder).get_trainloader(4)
    print('Start')
    for i, data_batched in enumerate(train_loader):
        print(data_batched['image'].size(0), data_batched['image'].size(1), data_batched['image'].size(2), data_batched['image'].size(3))
        print(data_batched['left'].size(0), data_batched['left'].size(1), data_batched['left'].size(2))
        print(data_batched['right'].size(0), data_batched['right'].size(1), data_batched['right'].size(2))
        if i > 5:
            break

test_train_loader()
