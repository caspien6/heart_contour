import os
import shutil

'''
This module deals with the dcm and con pairs.
Creates statistics and deletes files without pairs.
'''

def dcom_con_pairs(folder):
    '''
    folder - in the folder the subfolders are in the structure 
             provided by the anonymizer (studyID -> series -> imgs, contour.con)
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
            if exist_dcm and exist_con:
                num_pairs += 1
            elif exist_dcm:
                num_singledcm += 1
            elif exist_con:
                num_singlecon += 1
            else:
                num_strange += 1

    print("Number of files: %d"%num_all)
    print("Percentage of pairs: %.1f %%"%(num_pairs/num_all * 100.0))
    print("Percentage of single dcms: %.1f %%"%(num_singledcm/num_all * 100.0))
    print("Percentage of single cons: %.1f %%"%(num_singlecon/num_all * 100.0))
    print("Percentage of strange cases: %.1f %%"%(num_strange/num_all * 100.0))

#folder = "D:\AI\works\Heart\data\sa_all_5"
#dcom_con_pairs(folder)


def delete_unpaired(folder):
    '''
    folder - in the folder the subfolders are in the structure 
             provided by the anonymizer (studyID -> series -> imgs, contour.con)
    '''
    imgs = 'imgs'
    contour = 'contour.con'

    num_kept = 0.0
    num_all = 0.0

    for patient in os.listdir(folder):

        study_folder = os.path.join(folder, patient)
        
        for serial in os.listdir(study_folder):
            
            series_folder = os.path.join(study_folder, serial)
            exist_dcm = os.path.exists(os.path.join(series_folder, imgs))
            exist_con = os.path.exists(os.path.join(series_folder, contour))
            
            num_all += 1.0
            if not(exist_dcm and exist_con):
                shutil.rmtree(series_folder)
            else:
                num_kept += 1
    
        # delete empty directory
        if len(os.listdir(study_folder)) == 0:
            os.rmdir(study_folder)


    print("Kept files: %.1f %%"%(num_kept/num_all * 100.0))

folder = "D:\AI\works\Heart\data\sa_all_5"
delete_unpaired(folder)