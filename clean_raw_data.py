import pydicom as dicom
import os
import shutil
from multiprocessing import Process, Lock


'''
Reads in the origial raw measurements (*.dicom and *.con files).
Then the following operations are performed:
- anonimyzation
- choose the important images
'''
class CleanRawData:

    def __init__(self, root, target, thread_num=2, protocol="sBTFE_BH SA"):
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
        self.thread_num = thread_num
        self.protocol = protocol
    
    def create_folder_chunks(self):
        '''
        Splits the data into parts for multiprocessing.
        '''
        dcms_per_threads = [0] * self.thread_num
        cons_per_threads = [0] * self.thread_num

        base = self.root.split(os.sep)[-1]
        new_roots = [self.root.replace(base, 'root' + str(i)) for i in range(self.thread_num)]

        for path, _, files in os.walk(self.root): # iterate over all the folders

            num_files = len(files)
            files_per_thread = [num_files // self.thread_num] * self.thread_num
            for i in range(num_files % self.thread_num):
                files_per_thread[i] += 1
            
            start_idx, end_idx = 0, 0
            for i, new_root in enumerate(new_roots, 0): # splitting files into thread_num pieces
                start_idx = end_idx
                end_idx += files_per_thread[i]
                for j in range(start_idx, end_idx):

                    # counting files
                    if files[j].lower().endswith('.dcm'):
                        dcms_per_threads[i] += 1
                    elif files[j].lower().endswith('.con'):
                        cons_per_threads[i] += 1
                    else:
                        print('Warning: Unknown file format: %s at path: %s' % (files[j], path))
                    
                    # moving file into new root folder
                    name = files[j]
                    old_full_name = os.path.join(path, name)

                    # creating new folder structure
                    new_path = path.replace(self.root, new_root)
                    base_folder = ''
                    for folder in new_path.split(os.sep):
                        base_folder=os.path.join(base_folder, folder)
                        if not os.path.exists(base_folder):
                            os.mkdir(base_folder)
                    new_full_name = os.path.join(new_path, name)
                    shutil.move(old_full_name, new_full_name)
        
        self.new_roots = new_roots # saving root folder names
        self.file_statistic = (dcms_per_threads, cons_per_threads)
    
    def walk_dcm(self, lock, root, num_files):
        root_DCM = os.path.join(root, 'DCOMS')
        processed_files = 0
        for path, _, files in os.walk(root_DCM):
            for name in files:
                if name.lower().endswith('dcm'):
                    # filter: old path
                    old_path = self.__filter_dcm_oldpath(path, name)
                    
                    ds = dicom.dcmread(old_path)
                    if self.__filter_dcm_checkprotocol(ds):
                        # filter: new path
                        new_path = self.__filter_dcm_newpath(lock, ds, name)

                        # filter: anonym.
                        self.__filter_dcm_anonym(ds, old_path)

                        # filter: move to a new folder
                        self.__filter_dcm_move(old_path, new_path)
                    
                    else:
                        # filter: delete if not a picture according to the specified protocol
                        self.__filter_dcm_delete(old_path)
                    processed_files += 1 # a file was processed increase the number
                self.progress_bar(root, 'DCM', processed_files, num_files)


    def __filter_dcm_oldpath(self, path, name):
        return os.path.join(path, name)
    
    def __filter_dcm_checkprotocol(self, ds):
        protocolname = ds.ProtocolName
        seriesdescription = ds.SeriesDescription
        return protocolname.find(self.protocol) != -1 and seriesdescription.find(self.protocol) != -1
    
    def __filter_dcm_newpath(self, lock, ds, name):

        def create_folder(path):
            if not os.path.exists(path):
                lock.acquire()
                if not os.path.exists(path):
                    os.mkdir(path)
                lock.release()

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
        ds.InstitutionalDepartmentName = ""
        ds.StudyDate = ""
        ds.SeriesDate = ""
        ds.ContentDate = ""
        ds.AcquisitionDate = ""
        ds.InstanceCreationDate = ""
        ds.AccessionNumber = ""
        ds.StudyID = ""
        ds.PatientID = ""

        dicom.dcmwrite(old_path, ds)

    def __filter_dcm_move(self, old_path, new_path):
        shutil.move(old_path, new_path)

    def __filter_dcm_delete(self, old_path):
        os.remove(old_path)


    def walk_con(self, lock, root, num_files):
        root_CON = os.path.join(root, 'CONS')
        processed_files = 0
        for path, _, files in os.walk(root_CON):
            for name in files:
                if name.lower().endswith('con'):
                    # filter: old path
                    old_path = self.__filter_con_oldpath(path, name)
                    
                    # filter: new path
                    new_path = self.__filter_con_newpath(lock, old_path)

                    # filter: anonym.
                    self.__filter_con_anonym(old_path, new_path)

                    # filter: move to a new folder
                    self.__filter_con_deleteold(old_path)

                    processed_files += 1
                self.progress_bar(root, 'CON', processed_files, num_files)

    def __filter_con_oldpath(self, path, name):
        return os.path.join(path, name)
    
    def __filter_con_newpath(self, lock, old_path):

        def create_folder(path):
            if not os.path.exists(path):
                lock.acquire()
                if not os.path.exists(path):
                    os.mkdir(path)
                lock.release()

        tags = ['Study_id=', 'Series=']
        values = {}
        with open(old_path, 'rt', encoding='utf-8') as f:
            for line in f:
                for tag in tags:
                    idx = line.rstrip('\n').find(tag)
                    if idx != -1:
                        idx += len(tag)
                        values[tag] = line[idx:-1] # the final part of the line is the value we are interested in
                if len(values) == len(tags):
                    break
        
        temp = os.path.join(self.target, values[tags[0]])
        create_folder(temp)
        temp = os.path.join(temp, values[tags[1]])
        create_folder(temp)
        return os.path.join(temp, 'contour.con')

    def __filter_con_anonym(self, old_path, new_path):
        zeros_tags = [
           'Patient_name=',
           'Patient_gender=',
           'Birth_date=',
           'Study_date=',
           'Study_id=',
           'Patient_id='
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
    

    def clean_data(self, data):
        lock, root, file_statistic = data
        print('Process for %s was started.'%root)
        self.walk_dcm(lock, root, file_statistic[0])
        self.walk_con(lock, root, file_statistic[1])

    # function to write progress bar
    def progress_bar(self, chunk, process, processed_files, num_files):
        if processed_files % (num_files // 20 + 1) == 0 or processed_files == num_files:
           progress = int(processed_files / (num_files+1) * 100.0)
           print("Folder: %s, current phase: %s, Progress: [%d%%]" % (chunk, process, progress))

if __name__ == '__main__':
    source = 'D:\\AI\\works\\Heart\\code\\heart_contour\\root'
    target = 'E:\MLDATA\\SA_data\\SA_all'
    crd = CleanRawData(source, target, 4)
    crd.create_folder_chunks()
    print("Data were split up to parts. Start multiprocessing.")
    
    lock = Lock()
    for p in range(crd.thread_num):
        fn = crd.clean_data
        args = [(lock, crd.new_roots[i], (crd.file_statistic[0][i], crd.file_statistic[1][i])) for i in range(crd.thread_num)]
        Process(target=fn, args=(args[p],)).start()
