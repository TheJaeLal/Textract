import tensorflow as tf
import numpy as np
import shelve
import joblib
import os
from math import ceil
import time

import sys

#So that the below modules can be recognized by python interpreter
sys.path.insert(0,os.getcwd())

import neural_net_src.layers as layers
import neural_net_src.helper as helper
import neural_net_src.Image_Fetcher as Image_Fetcher
from neural_net_src.model import ANN_Model
from neural_net_src.Augment import valid_datagen
from neural_net_src.test_config import vocabulary,infer_batch_size,resume_epoch,img_height,img_width,model_dir,model_prefix
from neural_net_src.Arch import CNN

def get_text(input_dir_path):
    
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


    images = Image_Fetcher.fetch(input_dir_path)

    #Inputs for images, outputs for predictions, targets for labels
    infer_inputs = images
    infer_outputs = []
    infer_targets = []

    with tf.Session(graph = graph) as sess:

        saver = tf.train.Saver(max_to_keep=None)

        sess.run(tf.global_variables_initializer())

        timer  = 0

        saver.restore(sess, os.path.join(model_dir,model_prefix)+str(resume_epoch))

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

    #Prediction string (GOES TO FLASK SERVER)
    text = "\n".join(
          #All Sentences
            [ #List of sentences...
               "".join([vocabulary[char] for char in output[0][0][1]]).strip() for output in infer_outputs
            ]
    )

    print(text,file=sys.stderr)
    
    return text