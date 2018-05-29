#import tensorflow as tf
import keras
import numpy as np
from aug_config import *
from train_config import augment_data
import random
from scipy import ndimage as ndi
import math
#import joblib
import cv2

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

# def random_zoom(image,width):

#     """
#     If padding is too less --> Zoom Out than Zoom In
#     Otherwise --> Zoom In than Zoom Out
    
#     range -> []
    
#     375 -> content length
#     original_width * min_zoom_factor = 375
        
#     758 --> 350
#     max_zoom_factor = 715 / width
#     min_zoom_factor = 375 / width
    
    
#     """
    
#     max_zoom_factor = 720.0 / width
#     min_zoom_factor = 375.0 / width
    
#     #zoom_range = [min_zoom_factor, max_zoom_factor]
    
#     z = random.uniform(min_zoom_factor,max_zoom_factor)
    
#     #print("Before:",image.shape)
#     image = ndi.interpolation.zoom(image, z, mode = 'constant', cval=255.0)
#     #image = cv2.resize()
#     #print("Scaled:",image.shape)

#     new_height,new_width = image.shape
#     #print("new_height,new_width = ",new_height,new_width)
    
#     #Zoom In -> crop
#     if new_height > 114 or new_width > 758:
#         crop_top  = int(math.floor((new_height - 114)/2.0))
#         crop_left = int(math.floor((new_width - 758)/2.0))
 
#         #print("padd_top,padd_left = ",crop_top,crop_left)
 
#         image = image[crop_top:crop_top+114,crop_left:crop_left+758]
        
#     #Zoom Out --> padd
#     else:
#         new_image = np.zeros((114,758))
#         new_image.fill(255)

#         padd_top = int(math.floor((114 - new_height)/2.0))
#         padd_left = int(math.floor((758 - new_width)/2.0))
        
#         #print("padd_top,padd_left = ",padd_top,padd_left)
#         new_image[padd_top:padd_top+new_height,padd_left:padd_left+new_width] = image
          
#         image = new_image
        
#     #print("Zoomed:",image.shape)
    
#     return image

def cv2_clipped_zoom(img, zoom_factor):
    
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant',constant_values=255.0)
    assert result.shape[0] == height and result.shape[1] == width
    return result


def augment(image):
    
    #Extract width and height
    height = image[0][0][1]
        
    width = math.floor(image[0][0][2]/255.0 * 754.0)
    
    #print("******height,width = ",height,width)
    
    ##Sanity check...
    #if height!=image[0][1][1] or width!=math.floor(image[0][1][2]/255.0 * 754.0):
        #print("Gadbad hai....")
        #print(height,"!=",image[0][1][1])
        #print("or")
        #print(width,"!=",math.floor(image[0][1][2]/255.0 * 754.0))
    
    #print("Input image shape",image.shape)
    #Remove last 2 channels
    image = image[:,:,0]
    
    ##Convert image to int b/w 0 to 255
    image = image.astype(np.uint8)
    print("image.type =",image.dtype)
    
    #print("After removing last 2 channels",image.shape)
    
    #90% chances of augmentation....
    if random.randint(1,10) != 1:
        
        #Zoom
        if random.randint(1,1):
            
            #print("Performing Random Zoom")
            max_zoom_factor = 720.0 / width
            min_zoom_factor = 375.0 / width
            
            z = random.uniform(min_zoom_factor,max_zoom_factor)

            #print("Before:",image.shape)
            image = cv2_clipped_zoom(image,z)

        #print("Expanding image dims of shape",image.shape)
        #Required for keras preprocessing, channels as 1 for grayscale image
        image = np.expand_dims(image,axis=-1)

        #Rotate
        if random.randint(1,1):
            image = keras.preprocessing.image.random_rotation(image,2,
                                            row_axis=0,col_axis=1,channel_axis=2,
                                            fill_mode='constant',cval=255.0)

        #shear
        if random.randint(1,1):
            image = keras.preprocessing.image.random_shear(image,0.015,
                                            row_axis=0,col_axis=1,channel_axis=2,
                                            fill_mode='constant',cval=255.0)

        #height_shift  & width_shift
        if random.randint(1,1):   
    #         width_shift_range = 379.0/width - 0.5
    #         height_shift_range = 57.0/height - 0.5

            image = keras.preprocessing.image.random_shift(image,0.10,0.10,
                                            row_axis=0,col_axis=1,channel_axis=2,
                                            fill_mode='constant',cval=255.0)


    #Maybe causing lot of troubles for the Neural network to converge...(Maybe used as a fine-tuning step, later)
        # if random.randint(1,2) == 1:
        #     m = random.randint(baseline_upper_bound,baseline_lower_bound)
        #     out[m:m+baseline_width,:,:] = 255

        #Convert back to 3 channel..
        #print("squeezing image with shape",image.shape)
        
        image = np.squeeze(image,axis=-1)

    #Convert back to 3 channel image...
    channel_image = np.zeros((114,758,3),dtype=np.uint8)
    channel_image[:,:,0] = image

    #print("Final_Channel_image.shape = ",channel_image[:,:,0].shape)

    #Scale img_values b/w 0 to 1
    #image = channel_image / 255.0

    image = channel_image

    return image

if augment_data == True:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=augment)

else:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(data_format="channels_last")
    
valid_datagen = keras.preprocessing.image.ImageDataGenerator(data_format="channels_last")