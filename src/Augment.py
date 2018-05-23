import tensorflow as tf
from tensorflow import keras
import numpy as np
from aug_config import *
from train_config import augment_data
import random

def augment(image):
    
    out = image
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
    if random.randint(1,2) == 1:
        out = keras.preprocessing.image.random_rotation(out,max_rotation,row_axis=0,col_axis=1,channel_axis=2)

    #Zoom out
    if random.randint(1,2) == 1:
        out = keras.preprocessing.image.random_zoom(out,max_zoom,row_axis=0,col_axis=1,channel_axis=2)
    
    #Vertical Shift
#    if random.randint(1,2) == 1:
#        out = keras.preprocessing.image.random_shift(out,0,0.2,row_axis=0,col_axis=1,channel_axis=2)
 
#Maybe causing lot of troubles for the Neural network to converge...(Maybe used as a fine-tuning step, later)
    if random.randint(1,2) == 1:
        m = random.randint(baseline_upper_bound,baseline_lower_bound)
        out[m:m+baseline_width,:,:] = 255
        
    return out

if augment_data == True:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,preprocessing_function=augment)
else:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)