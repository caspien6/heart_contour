import os

def dcom_con_pairs(folder):
    '''
    folder - in the folder the subfolders are in the structure 
             provided by the anonymizer (studyID -> imgs, contour.con)
    '''
    imgs = 'imgs'
    contour = 'contour.con'

    num_pairs = 0.0
    num_singledcm = 0.0
    num_singlecon = 0.0
    num_strange = 0.0
    num_all = 0.0

    for patient in os.listdir(folder):

        study_folder = os.path.join(folder, patient)
        
        for serial in os.listdir(study_folder):
            
            series_folder = os.path.join(study_folder, serial)
            exist_dcm = os.path.exists(os.path.join(series_folder, imgs))
            exist_con = os.path.exists(os.path.join(series_folder, contour))
            
            num_all += 1.0
            if exist_dcm and exist_dcm:
                num_pairs += 1
            elif exist_dcm:
                num_singledcm += 1
            elif exist_con:
                num_singlecon += 1
            else:
                num_strange += 1

    print("Number of pairs: %.1f%%"%(num_pairs/num_all * 100.0))
    print("Number of single dcms: %.1f%%"%(num_singledcm/num_all * 100.0))
    print("Number of single cons: %.1f%%"%(num_singlecon/num_all * 100.0))
    print("Number of strange cases: %.1f%%"%(num_strange/num_all * 100.0))

folder = "D:\AI\works\Heart\data\SA_all_3"
dcom_con_pairs(folder)