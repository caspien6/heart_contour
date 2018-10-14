import numpy as np
import png
import pydicom
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
from os import listdir
from os.path import isfile, join
import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose
import roi.transform_classes as transform_classes

import roi.preprocess_img as preprocess_img


from roi.roi import RoiLearn
from roi.roi_dataset import RoiDataset
from roi.autoencoder import Autoencoder
from roi.preprocessor import Preprocessor
from PIL import Image
from roi.autoencoder_dataset import AEDataset



#preprocess_img.write_all_rectangle2file('/userhome/student/kede/heart_contour/heart_contour/sa_all_1/')


print(torch.device('cuda'))

project_root = '/userhome/student/kede/heart_contour/heart_contour'
csv_file = project_root + '/sa_all_1/rectangle.csv'
save_ae_weights_folder = project_root + '/ae_w.pth'
save_model_weights_folder = project_root + '/model_w.pth'

compose3 = Compose([transform_classes.StandardScale2(),transform_classes.ToTensor()])

ds2 = AEDataset(csv_file, compose3, sample_size = 1000)
roi = RoiLearn()
roi.build_ae()

crit = torch.nn.MSELoss(size_average = True)
opt = torch.optim.Adam(roi.autoencoder.parameters(), weight_decay = 0.0001 )

dataset_loader = torch.utils.data.DataLoader(ds2,batch_size=256, num_workers=0)
print('Learning ae weights starter.')
roi.learn_ae(dataset_loader, optimizer = opt, criterion = crit,  ep = 1000)

print('Learnging ae weights ready.')


roi.save_ae_weights(save_ae_weights_folder)

print('start learning')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.transforms import Compose
import transform_classes
import torch as th

th.cuda.set_device(0)

compose1 = Compose([transform_classes.ReScale64(),transform_classes.StandardScale(),transform_classes.ToTensor()])
compose2 = Compose([transform_classes.ReScale32(),transform_classes.ToTensor()])
ds = RoiDataset(csv_file, compose1, compose2,smpl = 0)

roi = RoiLearn()
roi.build_model()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(roi.model.parameters(),weight_decay = 0.0001)

dataset_loader = torch.utils.data.DataLoader(ds,batch_size=256, shuffle=False,num_workers=0)

roi.load_ae_weights(save_ae_weights_folder)
roi.ae_weights2model_feature_set()




import warnings
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
roi.learn_roi(dataset_loader, optimizer, criterion, ep = 1000)



roi.model.conv1.weight.requires_grad=True
roi.model.conv1.bias.requires_grad=True
roi.learn_roi(dataset_loader, optimizer, criterion, ep = 1000)
print('Save weights')

roi.save_model_weights(save_model_weights_folder)


print(torch.cuda.max_memory_allocated())
