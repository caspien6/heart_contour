import pydicom as dicom
import os
import shutil

'''
Reads in the origial raw measurements (*.dicom and *.con files).
Then the following operations are performed:
- anonimyzation
- choose the important images
'''
class CleanRawData:

    def __init__(self, root, target, protocol="sBTFE_BH SA"):
        '''
        root - the folder to read from
            structure: root - DCOMS - (patient folders with dcm files)
                            |
                            - CONS - (con files with the contours)
        target - the folder to save the results
        protocol - the required images (current project uses sBTFE_BH SA 
                but in case of necessaty it can be modified)
        '''
        self.root = root
        self.target = target
        self.protocol = protocol
    
    def walk_dcm(self):
        root_DCM = os.path.join(self.root, 'DCOMS')
        for path, _, files in os.walk(root_DCM):
            for i, name in enumerate(files, 1):
                if name.lower().endswith('dcm'):
                    # filter: old path
                    old_path = self.__filter_dcm_oldpath(path, name)
                    
                    ds = dicom.dcmread(old_path)
                    if self.__filter_dcm_checkprotocol(ds):
                        # filter: new path
                        new_path = self.__filter_dcm_newpath(ds, name)

                        # filter: anonym.
                        self.__filter_dcm_anonym(ds, old_path)

                        # filter: move to a new folder
                        self.__filter_dcm_move(old_path, new_path)
                    
                    else:
                        # filter: delete if not a picture according to the specified protocol
                        self.__filter_dcm_delete(old_path)
                self.progress_bar('DCM', i/len(files), path)


    def __filter_dcm_oldpath(self, path, name):
        return os.path.join(path, name)
    
    def __filter_dcm_checkprotocol(self, ds):
        protocolname = ds.ProtocolName
        seriesdescription = ds.SeriesDescription
        return protocolname.find(self.protocol) != -1 and seriesdescription.find(self.protocol) != -1
    
    def __filter_dcm_newpath(self, ds, name):

        def create_folder(path):
            if not os.path.exists(path):
                os.mkdir(path)

        study_id = ds.StudyID
        series_num = ds.SeriesNumber
        temp = os.path.join(self.target, str(study_id))
        create_folder(temp)
        temp = os.path.join(temp, str(series_num))
        create_folder(temp)
        temp = os.path.join(temp, 'imgs')
        create_folder(temp)
        return os.path.join(temp, name)

    def __filter_dcm_anonym(self, ds, old_path):
        ds.PatientName = ""
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.PatientAddress = ""
        ds.ReferringPhysicianName = ""
        ds.InstitutionName = ""
        ds.OperatorsName = ""

        dicom.dcmwrite(old_path, ds)

    def __filter_dcm_move(self, old_path, new_path):
        shutil.move(old_path, new_path)

    def __filter_dcm_delete(self, old_path):
        os.remove(old_path)


    def walk_con(self):

        root_CON = os.path.join(self.root, 'CONS')
        for path, _, files in os.walk(root_CON):
            for i, name in enumerate(files):
                if name.lower().endswith('con'):
                    # filter: old path
                    old_path = self.__filter_con_oldpath(path, name)
                    
                    # filter: new path
                    new_path = self.__filter_con_newpath(old_path, name)

                    # filter: anonym.
                    self.__filter_con_anonym(old_path, new_path)

                    # filter: move to a new folder
                    self.__filter_con_deleteold(old_path)
                self.progress_bar('CON', i/len(files), path)

    def __filter_con_oldpath(self, path, name):
        return os.path.join(path, name)
    
    def __filter_con_newpath(self, old_path, name):
        tags = ['Study_id=', 'Series=']
        values = []
        with open(old_path, 'rt', encoding='utf-8') as f:
            for line in f:
                for tag in tags:
                    idx = line.rstrip('\n').find(tag)
                    if idx != -1:
                        idx += len(tag)
                        values.append(line[idx:-1]) # the final part of the line is the value we are interested in
                if len(values) == len(tags):
                    break
        
        temp = os.path.join(self.target, values[0])
        temp = os.path.join(temp, values[1])
        return os.path.join(temp, name)

    def __filter_con_anonym(self, old_path, new_path):
        zeros_tags = [
           'Patient_name=',
           'Patient_gender=',
           'Birth_date='
        ]
        
        file_in = open(old_path, 'rt', encoding='utf-8')
        file_out = open(new_path, 'wt')

        for line in file_in:
            for tag in zeros_tags:
                idx = line.rstrip('\n').find(tag)
                if idx != -1:
                    line = tag + "0\n"
            file_out.write(line)
        
        file_in.close()
        file_out.close()

    def __filter_con_deleteold(self, old_path):
        os.remove(old_path)
    

    def clean_data(self):
        self.walk_dcm()
        self.walk_con()

    # function to write progress bar
    def progress_bar(self, process, progress, path):
        print("Current phase: %s, Progress: [%d%%] in Folder: %s ... \r"
        % (process, int(progress*100.0), path.encode('utf-8')), end='')


crd = CleanRawData("root", "target")
crd.clean_data()
