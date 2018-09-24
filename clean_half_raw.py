import os
import shutil


'''
A file for processing the first 200 data.
The dicom files were not preprocessed.
Reads in the origial raw measurements (*.dicom and *.con files).
Then the following operations are performed:
- anonimyzation
'''
class CleanRawData:

    def __init__(self, root, target, num_files):
        '''
        root - the folder to read from
            structure: root - DCOMS - (patient folders with dcm files)
                            |
                            - CONS - (con files with the contours)
        target - the folder to save the results
        '''
        self.root = root
        self.target = target
        self.num_files = num_files

    
    def walk_dcm(self, ds):
        root_DCM = os.path.join(self.root, 'DCOMS')
        processed_files = 0
        for path, _, files in os.walk(root_DCM):
            for name in files:
                if name.lower().endswith('dcm'):
                    # filter: old path
                    id = self.__filter_dcm_accessid(path)
                    old_path = self.__filter_dcm_oldpath(path, name)
                    new_path = self.__filter_dcm_newpath(ds, id, name)
                    self.__filter_dcm_move(old_path, new_path)
                    
                    processed_files += 1 # a file was processed increase the number
                self.progress_bar(self.root, 'DCM', processed_files, self.num_files)

    def __filter_dcm_accessid(self, path):
        parts = path.split(os.sep)
        for part in parts:
            if part.find('___AW') != -1:
                id = part.split('___AW')
                break
        return id[0]

    def __filter_dcm_oldpath(self, path, name):
        return os.path.join(path, name)
    
    def __filter_dcm_newpath(self, ds, id, name):

        def create_folder(path):
            if not os.path.exists(path):
                os.mkdir(path)

        study_id = ds[id]['study_id']
        series_num = ds[id]['series_num']
        temp = os.path.join(self.target, str(study_id))
        create_folder(temp)
        temp = os.path.join(temp, str(series_num))
        create_folder(temp)
        temp = os.path.join(temp, 'imgs')
        create_folder(temp)
        return os.path.join(temp, name)

    def __filter_dcm_move(self, old_path, new_path):
        shutil.move(old_path, new_path)

    def __filter_dcm_delete(self, old_path):
        os.remove(old_path)


    def walk_con(self):
        root_CON = os.path.join(self.root, 'CONS')
        processed_files = 0
        ds = {}
        for path, _, files in os.walk(root_CON):
            for name in files:
                if name.lower().endswith('con'):
                    # filter: old path
                    old_path = self.__filter_con_oldpath(path, name)
                    
                    # filter: accessing the id
                    id = self.__filter_con_accessid(name)

                    # filter: new path
                    new_path = self.__filter_con_newpath(ds, id, old_path)

                    # filter: anonym.
                    self.__filter_con_anonym(old_path, new_path)

                    # filter: move to a new folder
                    self.__filter_con_deleteold(old_path)

                    processed_files += 1
                self.progress_bar(self.root, 'CON', processed_files, self.num_files)
        return ds
    
    def __filter_con_accessid(self, name):
        id = name.split('.con')
        return id[0]

    def __filter_con_oldpath(self, path, name):
        return os.path.join(path, name)
    
    def __filter_con_newpath(self, ds, id, old_path):

        def create_folder(path):
            if not os.path.exists(path):
                os.mkdir(path)

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
        ds[id] = {'study_id': values[tags[0]], 'series_num': values[tags[1]]}
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
    

    def clean_data(self):
        print('Process for was started.')
        ds = self.walk_con()
        self.walk_dcm(ds)
        

    # function to write progress bar
    def progress_bar(self, chunk, process, processed_files, num_files):
        if processed_files % (num_files // 20 + 1) == 0 or processed_files == num_files:
           progress = int(processed_files / (num_files+1) * 100.0)
           print("Folder: %s, current phase: %s, Progress (number): [%d db]" % (chunk, process, progress))

if __name__ == '__main__':
    source = 'D:\AI\works\Heart\data\SA_original'
    target = 'D:\AI\works\Heart\data\SA_all_2'
    crd = CleanRawData(source, target, 0)
    crd.clean_data()