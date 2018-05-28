#import tensorflow as tf
from tensorflow import keras
import numpy as np
from aug_config import *
from train_config import augment_data
import random
from scipy import ndimage as ndi

# def erosion_dilation(image):
#     image = np.squeeze(image,axis=-1)
#     if random.randint(1,2) == 1:
#         #Erosion (thicken black foreground)
#         if random.randint(1,2) == 1:
#             image = ndi.grey_erosion(image,size=filter_size)
            
#         #Dilation (thin the black foreground)
#         else:
#             image = ndi.grey_erosion(image,size=filter_size)

#     image = np.expand_dims(image,axis=-1)
#     print(image.shape)
    
#     return image

# def augment(image):
    
#     out = image
    
#     #Rotation
#     if random.randint(1,2) == 1:
#         out = keras.preprocessing.image.random_rotation(out,max_rotation,row_axis=0,col_axis=1,channel_axis=2)

#     #Zoom out
#     if random.randint(1,2) == 1:
#         out = keras.preprocessing.image.random_zoom(out,max_zoom,row_axis=0,col_axis=1,channel_axis=2)
    
#     #Vertical Shift
# #    if random.randint(1,2) == 1:
# #        out = keras.preprocessing.image.random_shift(out,0,0.2,row_axis=0,col_axis=1,channel_axis=2)
 
# #Maybe causing lot of troubles for the Neural network to converge...(Maybe used as a fine-tuning step, later)
#     if random.randint(1,2) == 1:
#         m = random.randint(baseline_upper_bound,baseline_lower_bound)
#         out[m:m+baseline_width,:,:] = 255
        
#     return out

if augment_data == True:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
                       rotation_range=ror, width_shift_range=wsr, shear_range=shr,
                       height_shift_range=hsr, zoom_range=zr,rescale=1./255,
                       preprocessing_function=None,data_format="channels_last",fill_mode='constant',cval=255)

"""
featurewise_center=False,samplewise_center=False,
featurewise_std_normalization=False, samplewise_std_normalization=False,
zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0, 
height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, 
channel_shift_range=0.0, fill_mode='nearest', cval=0.0, 
horizontal_flip=False, vertical_flip=False, rescale=None, 
preprocessing_function=None, data_format=None, validation_split=0.0
"""
else:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,data_format="channels_last")
    
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,data_format="channels_last")