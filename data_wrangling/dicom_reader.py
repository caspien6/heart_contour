import pydicom as dicom
import numpy as np
import os


class DCMreader:

    def __init__(self, folder_name):

        self.num_slices = 0
        self.num_frames = 0

        images = []
        slice_locations = []

        dcm_files = os.listdir(folder_name)

        for file in dcm_files:

            if file.find('.dcm') != -1:
                temp_ds = dicom.dcmread(folder_name + file)
                images.append(temp_ds.pixel_array)
                slice_locations.append(temp_ds.SliceLocation)

        current_sl = -1
        frames = 0
        increasing = False
        indices = []
        for idx, slice_loc in enumerate(slice_locations):
            if slice_loc != current_sl:
                self.num_slices += 1
                self.num_frames = max(self.num_frames, frames)
                frames = 0
                indices.append(idx)

                if slice_loc > current_sl:
                    increasing = True
                else:
                    increasing = False
                
                current_sl = slice_loc
            frames += 1

        size_h, size_w = images[0].shape
        self.dcm_images = np.ones((self.num_slices, self.num_frames, size_h, size_w))
        self.dcm_slicelocations = np.ones((self.num_slices, self.num_frames, 1))

        for i in range(len(indices) - 1):

            for idx in range(indices[i], indices[i+1]):
                slice_idx = (i if increasing else (len(indices) - 1 - i))
                frame_idx = idx - indices[i]
                self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
                self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]

        for idx in range(indices[-1], len(images)):
            slice_idx = len(indices) - 1
            frame_idx = idx - indices[-1]
            self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
            self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]

        self.num_images = len(images)

    def get_image(self, slice, frame):
        return self.dcm_images[slice, frame, :, :]
    
    def get_slicelocation(self, slice, frame):
        return self.dcm_slicelocations[slice, frame]
