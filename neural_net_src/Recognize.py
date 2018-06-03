import tensorflow as tf
import numpy as np
import shelve
import joblib
import os
from math import ceil
import time

import sys

sys.path.insert(0, '/home/ubuntu/hcr-ann/src')

import layers, helper, Image_Fetcher
from model import ANN_Model
from Augment import valid_datagen
from test_config import vocabulary,infer_batch_size,mount_point,resume_epoch,img_height,img_width,model_dir,model_prefix
from Arch import CNN

def get_text(input_dir):
    
    #Importing model parameters
    model_params = ANN_Model()

    graph = model_params[0]
    dropout_lstm = model_params[1]
    dropout_fc = model_params[2]
    inputs = model_params[3]
    time_steps = model_params[4]
    targets = model_params[5]
    loss = model_params[6]
    train = model_params[7]
    decoded = model_params[8]
    label_error_rate = model_params[9]
    seq_len = model_params[10]
    is_training = model_params[11]
    conv_dropout = model_params[12]
    gradients = model_params[13]
    interim_dropout = model_params[14]

    #Generating images
    # valid_generator = valid_datagen.flow_from_directory( os.path.join(mount_point,input_dir),
    #                                             target_size=(img_height,img_width), color_mode='grayscale',
    #                                             batch_size = infer_batch_size, shuffle=False )

    images = Image_Fetcher.fetch(os.path.join(mount_point,input_dir))

    #Inputs for images, outputs for predictions, targets for labels
    infer_inputs = images
    infer_outputs = []
    infer_targets = []

    with tf.Session(graph = graph) as sess:

        saver = tf.train.Saver(max_to_keep=None)

        sess.run(tf.global_variables_initializer())

        timer  = 0

        saver.restore(sess, os.path.join(mount_point,model_dir,model_prefix)+str(resume_epoch))

        start_time = time.time()

        count = 0

        for x in images:

        #Placeholder requires no_of_channels at the end, and at 0th Index..
            x = np.expand_dims(x,axis=-1)
            x = np.expand_dims(x,axis=0)


            feed = {
                         inputs:x,time_steps:[seq_len],
                         conv_dropout:[1]*len(CNN),
                         dropout_fc:1,dropout_lstm:1,interim_dropout:1,
                         is_training:False
                    }

            d = sess.run([decoded],feed_dict=feed)
            infer_outputs.append(d)
            count+=1

    #infer_outputs[0][0][0][1]

    #Prediction string (PROBABLY GOES TO FLASK SERVER)
    text = "\n".join(
          #All Sentences
            [ #List of sentences...
               "".join([vocabulary[char] for char in output[0][0][1]]).strip() for output in infer_outputs
            ]
    )

    print(text,file=sys.stderr)
    
    return text


# ****CODE TO REVIEW*****

# #Number of line images. ***Considering [0][0] is [batch][img_number]. Might have to change***
# number_of_images = infer_inputs.shape[1]

# #Concatenate prediction of all lines
# #Not sure of index number for image number. Check that after running the code.
# #Check syntax also
# transcription = ""
# for img_number in range(0,number_of_images):
#     transcription = transcription+"\n".join([vocabulary[char] for char in infer_outputs[0][img_number][0][1]])

# print(transcription)


# #Validating input image with prediction
# original_img = infer_inputs[0][0]
# original_img = original_img.reshape(original_img.shape[:2])

# #original_img.shape

# plt.imsave('test_input.jpg',original_img,cmap='gray',format='jpg')