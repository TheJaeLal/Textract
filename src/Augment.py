import tensorflow as tf
from tensorflow import keras
import numpy as np
from aug_config import *
import random

def augment(image):
    n = random.randint(1,8)
    #No Augmentation
    if n==1 or n==3 or n==5 or n==7:
        return image
    
#    #Dilation
#     elif n==2:
#         image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
#         dilout = tf.nn.dilation2d(image,filter=myfilter,strides=[1,1,1,1],rates=[1,1,1,1],padding='SAME')
#         with tf.Session() as sess:
#             out = dilout.eval(session=sess)
#         out = out.reshape(out.shape[1],out.shape[2],out.shape[3])
#       
#     #Erosion 
#     elif n==4:
#         image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
#         eroded = tf.nn.erosion2d(image,kernel=myfilter,strides=[1,1,1,1],rates=[1,1,1,1],padding='SAME')
#         with tf.Session() as sess:
#             out = eroded.eval(session=sess)
#         out = out.reshape(out.shape[1],out.shape[2],out.shape[3])
    
    #Rotation
    elif n==6:
        out = keras.preprocessing.image.random_rotation(image,max_rotation,row_axis=0,col_axis=1,channel_axis=2)

    #Zoom out
    elif n==4:
        out = keras.preprocessing.image.random_zoom(image,max_zoom,row_axis=0,col_axis=1,channel_axis=2)
    
    #Vertical Shift
    elif n==2:
        out = keras.preprocessing.image.random_shift(image,0,0.2,row_axis=0,col_axis=1,channel_axis=2)
    
    else:
        m = random.randint(baseline_upper_bound,baseline_lower_bound)
        image[m:m+baseline_width,:,:] = 255
        out = image
    return out

train_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=augment)
#train_generator = train_datagen.flow(
#                                        directory=img_folder_path,
#                                        target_size=(104,688),
#                                        color_mode='grayscale',class_mode=None,
#                                       batch_size=7,save_to_dir=out_folder_path,
#                                        save_prefix='transformed'
#                                    )