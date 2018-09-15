
# coding: utf-8

# In[162]:


import numpy as np
import png
import pydicom
from sklearn.preprocessing import normalize

from os import listdir
from os.path import isfile, join
import torch
import torch.nn as nn


# In[19]:




onlyfiles = [f for f in listdir('root/DCOMS') if isfile(join('root/DCOMS', f))]

for filename in onlyfiles:
    ds = pydicom.dcmread('root/DCOMS/' + filename)

    shape = ds.pixel_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    
    output_filename = filename.rsplit('.', 1)[0]
    
    # Write the PNG file
    with open('pngs/' + output_filename+'.png', 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)




# In[ ]:


model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(50):
    # Forward Propagation
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()


# In[222]:


from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )


# In[309]:


class RoiLearn:
    def __init__(self):
        torch.manual_seed(44)
        self.conv1 = nn.Conv2d(1,1, (1,1))
        self.sigmoid = nn.Sigmoid()
        
    def build_model(self):
        self.model = nn.Sequential(self.conv1,
                            self.sigmoid)
        self.model = self.model.double()
    
    def load_and_prepare_image(self, path):
        p = load_image(path)
        normed_matrix = normalize(p)
        normed_matrix = np.expand_dims(normed_matrix, 0)
        normed_matrix = np.expand_dims(normed_matrix, 0)
        self.x = normed_matrix
        self.x = torch.from_numpy(self.x)
    
    def propagate(self):
        return self.model(self.x)
    
    def save_image( npdata, outfilename ) :
        img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
        img.save( outfilename )


# In[310]:


pat = 'pngs/1.3.46.670589.11.22133.5.0.4480.2017032208061921840.png'
roi = RoiLearn()


# In[311]:


roi.build_model()


# In[312]:


roi.load_and_prepare_image(pat)
y_pred = roi.propagate().detach().numpy()
save_image(y_pred[0,0,:,:]*255, 'pred.png')

