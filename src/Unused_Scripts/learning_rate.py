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

def find(ler_range,multiplier,update_frequency,resume_epoch=0):

    min_ler,max_ler = ler_range
    
    mount_point = '../'
    
    #Load training data
    train_array,train_label = load_train(mount_point)
    graph, dropout_lstm,dropout_fc,inputs,time_steps,targets,loss,train,decoded,label_error_rate,seq_len,is_training, conv_dropout,learning_rate = ler_model.model()
    
    train_generator = train_datagen.flow(train_array,train_label,batch_size)

    with tf.Session(graph = graph) as sess:
        
        saver = tf.train.Saver(max_to_keep=None)

        sess.run(tf.global_variables_initializer())

        losses = []
        learn_rates = []

        alpha = min_ler

        batch_count = 0
        
        #Resume training from resume_epoch
        if resume_epoch != 0:
            saver.restore(sess, mount_point+'saved_models/cnn_lstm_fc_'+str(resume_epoch))
        
        avg_loss = 0.0
        
        #Mini Batch loop
        for x,y in train_generator:

            batch_count += 1
            
            if alpha > max_ler:
                break

            #Need this as actual batch size may be less for the last mini-batch 
            #if num of samples is not exactly divisible.
            actual_batch_size = x.shape[0]

            sparse_y = helper._batch_y(y,vocabulary)

            feed_train = {
                             inputs:x,targets:sparse_y,
                             time_steps:np.array([seq_len]*actual_batch_size),
                             conv_dropout:[1,1,0.8,0.8,0.8],dropout_fc:1,dropout_lstm:0.5,
                             learning_rate:alpha,is_training:True
                        }
            
            _,cost_val = sess.run([train,loss],feed_dict= feed_train)

            avg_loss += cost_val
            
            if (batch_count % update_frequency == 0 ):
                learn_rates.append(alpha)
                alpha *= multiplier
                avg_loss = avg_loss / update_frequency
                losses.append(avg_loss)
                avg_loss = 0.0
        
        return losses,learn_rates