
import tensorflow as tf
import numpy as np
import joblib
import shelve

import ler_model
from Augment import train_datagen
from config import vocabulary, batch_size
import helper

def load_train(mount_point):

    with shelve.open(mount_point+'IAM_Data','c') as shelf:
        train_label = shelf['train_label']

    train_array = joblib.load(mount_point+'data/train_array')

    return train_array,train_label

def find(min_ler,max_ler,multiplier):

    mount_point = '../'
    
    #Load training data
    train_array,train_label = load_train(mount_point)
    
    params = ler_model.model()
    
    graph = params[0]; dropout_conv = params[1]; dropout_lstm = params[2]
    dropout_fc = params[3]; inputs = params[4]; time_steps = params[5]
    targets = params[6]; loss = params[7]; train = params[8]
    decoded = params[9]; label_error_rate = params[10]; seq_len = params[11]
    learning_rate = params[12]
    
    train_generator = train_datagen.flow(train_array,train_label,batch_size)

    with tf.Session(graph = graph) as sess:
    
        sess.run(tf.global_variables_initializer())

        losses = []
        learn_rates = []

        alpha = min_ler

        #Mini Batch loop
        for x,y in train_generator:

            if alpha > max_ler:
                break

            #Need this as actual batch size may be less for the last mini-batch 
            #if num of samples is not exactly divisible.
            actual_batch_size = x.shape[0]

            sparse_y = helper._batch_y(y,vocabulary)

            feed_train = {
                         inputs:x,targets:sparse_y,
                         time_steps:np.array([seq_len]*actual_batch_size),
                         dropout_conv:0.5,dropout_fc:0.5,dropout_lstm:0.5,
                         learning_rate:alpha
                    }

            _,cost_val = sess.run([train,loss],feed_dict= feed_train)

            losses.append(cost_val)
            learn_rates.append(alpha)

            alpha *= multiplier
            
        
        return losses,learn_rates