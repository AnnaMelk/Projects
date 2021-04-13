#!/usr/bin/env python
# coding: utf-8

# # Transformations #

# ## 2D ##

# In[21]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 


# In[2]:


from scipy.ndimage import rotate
from scipy.interpolate import RegularGridInterpolator

def randomrotation(image, degrees, prob):
    angle = np.random.uniform(-degrees, degrees)
    np_image = image.detach().numpy()[0,0,:,:]
    
    dim1, dim2 = np_image.shape
    x = np.arange(0, dim1)
    y = np.arange(0, dim2)
    
    # make the interpolator
    fn = RegularGridInterpolator((x, y), np_image)
    rand = np.random.uniform()
    output = np.zeros_like(np_image)

    if rand <= prob:
        #rotate the image and find the outside boudaries 
        output = sp.ndimage.interpolation.rotate(np_image, angle, reshape=False)
        xdim, ydim = np.where(output == 0)
        
        #replace the boundary with the values from the original image
        pts0 = np.vstack((xdim, ydim)).T
        interp_points = fn(pts0)
        output[xdim, ydim] = interp_points

        output = np.expand_dims(output, axis=0)
        output = np.expand_dims(output, axis=0)
        augmented = torch.Tensor(output)
    else:
        augmented = image

    return augmented


# In[3]:


def colorjitter(image, brightness=0):

    np_image = image.detach().numpy()
    layer = np.squeeze(np_image)
    layer = torch.from_numpy(layer)

    brightness_factor = torch.tensor(1.0).uniform_(0, brightness).item()
    layer = torchvision.transforms.functional.adjust_brightness(image, brightness_factor)
    layer = layer.detach().numpy()
    augmented = torch.Tensor(layer)

    return augmented


# ## 3D ##

# In[4]:


class Pad:

    def __init__(self, axis=0, ps=3):
        self.axis = axis
        self.ps = ps

    def __call__(self, image):
        augmented = torch.nn.functional.pad(image, (self.ps, self.ps, self.ps, self.ps, self.ps, self.ps), 
                                            mode='replicate')
 
        return augmented


# In[5]:


class Crop:

    def __init__(self, axis=0, ps=3):
        self.axis = axis
        self.ps = ps

    def __call__(self, image):
        augmented = image[:, :, self.ps:-self.ps, self.ps:-self.ps, self.ps:-self.ps]
        
        return augmented


# In[6]:


from scipy.ndimage import rotate
from scipy.interpolate import RegularGridInterpolator

# instead of using the interpolator here, could also fill the boundaries with 
# mode parameters in sp.ndimage.interpolation.rotate function
# options: {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}

class RandomRotation:

    def __init__(self, degrees=0, prob=0.5, axis=0):
        self.degrees = degrees
        self.axis = axis
        self.prob = prob
        self.angle = np.random.uniform(-self.degrees, self.degrees)
        
    def __call__(self, image):
        np_image = image.detach().numpy()
        np_image = np.squeeze(np_image)
        dim1, dim2, dim3 = np_image.shape
        x = np.arange(0, dim1)
        y = np.arange(0, dim2)
        z = np.arange(0, dim3)
        # make the interpolator
        fn = RegularGridInterpolator((x, y, z), np_image)
        rand = np.random.uniform()
        output = np.zeros_like(np_image)
        
        #rotate depending on the axis given
        if rand <= self.prob:
            if self.axis == 0:
                output = sp.ndimage.interpolation.rotate(np_image, self.angle, axes=(0,2), reshape=False)
            elif self.axis == 1:
                output = sp.ndimage.interpolation.rotate(np_image, self.angle, axes=(0,1), reshape=False)
            elif self.axis == 2:
                output = sp.ndimage.interpolation.rotate(np_image, self.angle, axes=(1,2), reshape=False)
            
            #find the boundaries and replace them
            xdim, ydim, zdim = np.where(output == 0)
            pts0 = np.vstack((xdim, ydim, zdim)).T
            interp_points = fn(pts0)
            output[xdim, ydim,zdim] = interp_points
            
            output = np.expand_dims(output, axis=0)
            output = np.expand_dims(output, axis=0)
            augmented = torch.Tensor(output)
        else:
            augmented = image
        
        return augmented


# In[7]:


class FlipTorch:

    def __init__(self, axis=0, prob=0.5):
        self.axis = axis
        self.prob = prob

    def __call__(self, image):
        rand = np.random.uniform()
        if rand <= self.prob:
            if self.axis == 0:
                augmented = torch.flip(image, dims=(0, 2))
            elif self.axis == 1:
                augmented = torch.flip(image, dims=(0, 1))
            elif self.axis == 2:
                augmented = torch.flip(image, dims=(1, 2))
        else:
            augmented = image
        return augmented


# In[8]:


import torch.nn.functional as F

class RandomResizedCrop:

    def __init__(self, scale):
        #a multiplier (integer)
        self.scale = scale

    def __call__(self, image):
        image_np = image.detach().numpy()
        image_np = np.squeeze(image_np)
        height, width, depth = image_np.shape
        volume = height * width * depth
        ratio = (height/width, height/width)
        range_scale = (self.scale, 1.00)
        #pick a random scale
        target_volume = volume * torch.empty(1).uniform_(range_scale[0], range_scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        #keep the ratio of the original image
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        #find new dimensions with the new volume and ratio 
        h = int(round((target_volume * aspect_ratio)** (1./3.)))
        w = int(round((target_volume / aspect_ratio)** (1./3.)))
        d = int(round((target_volume * aspect_ratio)** (1./3.)))

        #for new dimensions, pick a random value at which the original image will be cut
        if 0 < w <= width and 0 < h <= height and 0 < d <= depth:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            z = torch.randint(0, depth - d + 1, size=(1,)).item()
        
        #crop and then interpolate the cropped image to the original size
        crop = image_np[i:np.min([-1,-(height-h-i)]), j:np.min([-1,(width-w-j)]), z:np.min([-1,(depth-d-z)])]
        output = np.expand_dims(crop, axis=0)
        output = np.expand_dims(output, axis=0)
        output = torch.Tensor(output)
        rescaled = F.interpolate(output, size=(height, width, depth), mode='trilinear', align_corners=False) 
        
        return rescaled


# In[ ]:




