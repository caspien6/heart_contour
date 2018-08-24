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
        target - the folder to which copy the processed files
            structure: target - patient_1 - patient_dcms (folder for dcm files)
                              |           |         
                              |           - .con file
                              - patient_2 ...
        protocol - the required images (current project uses sBTFE_BH SA 
                but in case of necessaty it can be modified)
        '''
        self.root = root
        self.target = target
        self.protocol = protocol

    def create_target_folder(self):
        if os.path.exists(self.target):
            shutil.rmtree(self.target)
        os.mkdir(self.target)
        print("Target folder was created!")
    
    def find_dcoms_con(self):
        root_dcms = os.path.join(self.root, 'DCOMS')
        root_cons = os.path.join(self.root, 'CONS')

        dcm_dirs = os.listdir(root_dcms)
        con_files = os.listdir(root_cons)
        self.num_patients = len(dcm_dirs)

        # creating a dictionary for cons with series number key
        studyid_con_dict = {}
        for i, conf in enumerate(con_files, 1):
            # read from con file (patientID, series number)

            self.progress_bar("indexing *.con files", i, 1.0)
        print("")

        # creating a list with patients and corresponding dcm files
        patients_dcms = []
        for id, patient_dir in enumerate(dcm_dirs, 1):
            patient = {'dcm_names': [], 'id':id}
            for path, _, files in os.walk(os.path.join(root_dcms, patient_dir)):
                for j, name in enumerate(files, 1):
                    if name.lower().endswith('dcm'):
                        file_path = os.path.join(path, name)
                        patient['dcm_names'].append(file_path)
                    self.progress_bar("indexing *.dcm files", id, j/len(files))
            patients_dcms.append(patient)
        print("")
        
        self.patients_dcms = patients_dcms 
        self.studyid_con_dict = studyid_con_dict

        print("Dcm files and con files were found!")       

    def select_images(self):
        # tags for the selection
        tags = ["ProtocolName", "SeriesDescription", "StudyID", "SeriesNumber"]

        patients_sa_dcms = []
        for pd in self.patients_dcms:
            new_pd = {'dcm_names': [], 'id':pd['id']}
            for j, dcm in enumerate(pd['dcm_names'], 1):
                ds = dicom.dcmread(dcm, specific_tags=tags)
                protocolname = ds.ProtocolName
                seriesdescription = ds.SeriesDescription
                if (protocolname.find(self.protocol) != -1 
                    and seriesdescription.find(self.protocol) != -1 
                    and int(ds.SeriesNumber) in self.series_in_cons):
                    new_pd['dcm_names'].append(dcm)
                    if 'studyid' in new_pd.keys() and 'seriesnumber' in new_pd.keys():
                        assert new_pd['studyid'] == ds.StudyID
                        assert new_pd['seriesnumber'] != ds.SeriesNumber
                    else:
                        new_pd['studyid'] = ds.StudyID
                        new_pd['seriesnumber'] = ds.SeriesNumber

                self.progress_bar("selecting SA images", pd['id'], j/len(pd['dcm_names']))
            patients_sa_dcms.append(new_pd)
        
        self.patients_sa_dcms = patients_sa_dcms
        print("")
        print("Images with the specified protocol were chosen!")
    
    def move_to_target(self):
        
        for i, patient_dcms in enumerate(self.patients_sa_dcms, 1):
            path = os.path.join(self.target, "patient"+str(i))
            path_sub = os.path.join(path, "patient_dcms")
            os.mkdir(path)
            os.mkdir(path_sub)

            for j, dcm_file in enumerate(patient_dcms['dcm_names'], 1):
                new_file_name = os.path.basename(dcm_file)
                new_path = os.path.join(path_sub, new_file_name)
                shutil.move(dcm_file, new_path)

                self.progress_bar("saving images to target folder", i, j/len(patient_dcms['dcm_names']))
            
            con_path = self.studyid_con_dict[patient_dcms['studyid']]
            shutil.move(con_path, os.path.join(path, str(i)+'.con'))
        
        print("")
        print("Images were moved to target!")
    
    def anonimyzation(self):
        
        for i, (path, _, files) in enumerate(os.walk(self.target), 1):
            for j, name in enumerate(files, 1):
                if name.lower().endswith('dcm'):
                    dcm_path = os.path.join(path, name)
                    ds = dicom.dcmread(dcm_path)
                    ds.PatientName = ""
                    ds.PatientBirthDate = ""
                    ds.PatientSex = ""
                    ds.PatientAddress = ""
                    ds.StudyDate = ""
                    ds.StudyTime = ""
                    ds.SeriesData = ""
                    ds.SeriesTime = ""
                    ds.ReferringPhysicianName = ""
                    ds.InstitutionName = ""
                    ds.OperatorsName = ""

                    dicom.dcmwrite(dcm_path, ds)

                self.progress_bar("anonymizing", i, j/len(files))

        print("")
        print('Anonymization done!')

    def clean_data(self):
        self.create_target_folder()
        self.find_dcoms_con()
        self.select_images()
        self.move_to_target()
        self.anonimyzation()

    # function to write progress bar
    def progress_bar(self, process, patient, progress):
        print("Current phase: %s, Patient: %d / %d, Progress: [%d%%] ...\r"
        % (process, patient, self.num_patients, int(progress*100.0)), end='')


crd = CleanRawData("root", "target")
crd.clean_data()
