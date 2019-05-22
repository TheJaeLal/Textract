import tensorflow as tf
import numpy as np
import shelve
import joblib
from math import ceil
import time
import os

import neural_net_src.model as model
from neural_net_src.Augment import valid_datagen
import neural_net_src.helper as helper
from neural_net_src.test_config import vocabulary, test_batch_size, resume_epoch, model_dir, model_prefix
from neural_net_src.Arch import CNN
import neural_net_src.layers as layers


# ## Load Test Labels and Image-Array

with shelve.open('Metadata','c') as shelf:
    test_label = shelf['test_label']
    
test_array = joblib.load(os.path.join('data','test_array'))

graph,dropout_lstm,dropout_fc,inputs,time_steps,targets,loss,train,decoded,label_error_rate,seq_len,is_training,conv_dropout,gradients,interim_dropout = model.model()

# ## Test Parameters

test_generator = valid_datagen.flow(test_array,test_label,test_batch_size)

num_test_samples = test_array.shape[0]

num_batches = int(ceil(num_test_samples/test_batch_size))

with tf.Session(graph = graph) as sess:
    
    saver = tf.train.Saver(max_to_keep=None)
    
    sess.run(tf.global_variables_initializer())
    
    checkpoint = False

    timer  = 0
    
    #Resume training from resume_epoch
    if resume_epoch != 0:
        saver.restore(sess, os.path.join(model_dir, model_prefix + str(resume_epoch)))

    start_time = time.time()
    
    test_loss,ler = 0.0,0.0

    count = 0

    for xt, yt in test_generator: 

        if count == num_batches:
            break

        yt,widths = np.hsplit(yt,2)

        widths = np.squeeze(widths,axis=1)
        yt = np.squeeze(yt,axis=1)

        widths = [layers.calc_out_dims(CNN,0,int(width))[1] for width in widths]
        widths = np.array(widths)

        test_size = xt.shape[0]
        sparse_targets = helper._batch_y(yt,vocabulary)

        feed_test = {
                     inputs:xt,targets:sparse_targets,
                     time_steps:widths,
                     conv_dropout:[1]*len(CNN),dropout_fc:1,dropout_lstm:1,
                     interim_dropout:1,is_training:False
            }

        t_loss_val, t_ler= sess.run([loss,label_error_rate],feed_dict = feed_test)

        test_loss += t_loss_val
        ler += t_ler
        count+=1

    test_loss /= num_batches
    ler /= num_batches

    end_time = time.time()
    time_taken = end_time - start_time

    print("Test_loss:{:.2f}, Accuracy:{:.6f}, {:.2f} sec.\n".format(test_loss,(1-ler)*100.0,time_taken))
        
    with open('progress.csv','a') as f:
        f.write("Test_loss:{:.2f}, Accuracy:{:.6f}, {:.2f} sec.\n".format(test_loss,(1-ler)*100.0,time_taken))