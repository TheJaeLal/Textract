#import tensorflow as tf
import keras
import numpy as np
from aug_config import *
from train_config import augment_data
import random
from scipy import ndimage as ndi
import math

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

def random_zoom(image,width):

    """
    If padding is too less --> Zoom Out than Zoom In
    Otherwise --> Zoom In than Zoom Out
    
    range -> []
    
    375 -> content length
    original_width * min_zoom_factor = 375
        
    758 --> 350
    max_zoom_factor = 715 / width
    min_zoom_factor = 375 / width
    
    
    """
    
    max_zoom_factor = 720.0 / width
    min_zoom_factor = 375.0 / width
    
    #zoom_range = [min_zoom_factor, max_zoom_factor]
    
    z = random.uniform(min_zoom_factor,max_zoom_factor)
    
    #print("Before:",image.shape)
    image = ndi.interpolation.zoom(image, z, mode = 'constant', cval=255.0)
    #print("Scaled:",image.shape)

    new_height,new_width = image.shape
    
    #Zoom In -> crop
    if new_height > 114:
        crop_top  = int(math.floor((new_height - 114)/2.0))
        crop_left = int(math.floor((new_width - 758)/2.0))
        
        image = image[crop_top:crop_top+114,crop_left:crop_left+758]
        
    #Zoom Out --> padd
    else:
        new_image = np.zeros((114,758))
        new_image.fill(255)

        padd_top = int(math.floor((114 - new_height)/2.0))
        padd_left = int(math.floor((758 - new_width)/2.0))
        
        new_image[padd_top:padd_top+new_height,padd_left:padd_left+new_width] = image
          
        image = new_image
        
    #print("Zoomed:",image.shape)
    
    return image

def augment(image):
    
    #Extract width and height
    height = image[0][0][1]
    
    width = math.floor(image[0][0][2]/255.0 * 754.0)
    
    print("******height,width = ",height,width)
    
    ##Sanity check...
    if height!=image[0][1][1] or width!=math.floor(image[0][1][2]/255.0 * 754.0):
        print("Gadbad hai....")
        print(height,"!=",image[0][1][1])
        print("or")
        print(width,"!=",math.floor(image[0][1][2]/255.0 * 754.0))
        
    #Remove last 2 channels
    image = image[:,:,0]
    
    if random.randint(1,2):
        image = random_zoom(image,width)
    
    image = np.expand_dims(image,axis=-1)
    
    #Rotate
    if random.randint(1,2):
        image = keras.preprocessing.image.random_rotation(image,2,
                                        row_axis=0,col_axis=1,channel_axis=2,
                                        fill_mode='constant',cval=255.0)
    
    #shear
    if random.randint(1,2):
        image = keras.preprocessing.image.random_shear(image,0.015,
                                        row_axis=0,col_axis=1,channel_axis=2,
                                        fill_mode='constant',cval=255.0)
    
    #height_shift  & width_shift
    if random.randint(1,2):   
        image = keras.preprocessing.image.random_shift(image,0.10,0.2,
                                        row_axis=0,col_axis=1,channel_axis=2,
                                        fill_mode='constant',cval=255.0)
    
    #Rotation
    # if random.randint(1,2) == 1:
    #     out = 

    #Zoom out
    # if random.randint(1,2) == 1:
    #     out = keras.preprocessing.image.random_zoom(out,max_zoom,row_axis=0,col_axis=1,channel_axis=2)
    
    #Vertical Shift
#    if random.randint(1,2) == 1:
#        out = keras.preprocessing.image.random_shift(out,0,0.2,row_axis=0,col_axis=1,channel_axis=2)
 
#Maybe causing lot of troubles for the Neural network to converge...(Maybe used as a fine-tuning step, later)
    # if random.randint(1,2) == 1:
    #     m = random.randint(baseline_upper_bound,baseline_lower_bound)
    #     out[m:m+baseline_width,:,:] = 255
    
    #Convert back to 3 channel..
    image = np.squeeze(image,axis=-1)
    
    channel_image = np.zeros((114,758,3))
    channel_image[:,:,0] = image
    
    print("Final_Channel_image.shape = ",channel_image[:,:,0].shape)
    
    #Scale img_values b/w 0 to 1
    #image = channel_image / 255.0
    image = channel_image
    
    return image

if augment_data == True:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=augment)

else:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(data_format="channels_last")
    
valid_datagen = keras.preprocessing.image.ImageDataGenerator(data_format="channels_last")