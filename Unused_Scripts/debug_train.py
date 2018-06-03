
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shelve
import joblib
from math import ceil
import time
from tensorflow.python import debug as tf_debug

import model
from Augment import train_datagen, valid_datagen
import helper
from config import vocabulary,batch_size,valid_batch_size,n_epochs,resume_epoch,save_epoch


mount_point = '../'

with shelve.open(mount_point+'IAM_Data','c') as shelf:
    train_label = shelf['train_label']
    valid_label = shelf['valid_label']


train_array = joblib.load(mount_point+'data/train_array')

valid_array = joblib.load(mount_point+'data/valid_array')



# In[2]:


graph,dropout_conv,dropout_lstm,dropout_fc,inputs,time_steps,targets,loss,train,decoded,label_error_rate,seq_len,is_training= model.model()

# !rm -rf ../Augmented/*

train_generator = train_datagen.flow(train_array,train_label,batch_size)
valid_generator = valid_datagen.flow(valid_array,valid_label,valid_batch_size)


num_training_samples = train_array.shape[0]
num_valid_samples = valid_array.shape[0]


num_batches = int(ceil(num_training_samples/batch_size))
num_vbatches = int(ceil(num_valid_samples/valid_batch_size))


# In[ ]:


with tf.Session(graph = graph) as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.add_tensor_filter("has_inf_or_nan",tf_debug.has_inf_or_nan)
 
    saver = tf.train.Saver(max_to_keep=None)
    
    sess.run(tf.global_variables_initializer())
    checkpoint = False
    timer  = 0

    #Resume training from resume_epoch
    if resume_epoch != 0:
        saver.restore(sess, mount_point+'saved_models/cnn_lstm_fc_'+str(resume_epoch))
    
    #Epoch Loop
    for e in range(resume_epoch,n_epochs):
        start_time = time.time()
        
        train_loss = 0.0    
        count = 0
        
#         print('Training loop')
        #Mini Batch loop
        for x,y in train_generator:
            
#             plt.imshow(x[0].reshape(x[0].shape[:2]),cmap='gray')
#             print(y[0])
            
            actual_batch_size = x.shape[0]
            
#             print('Train Minibatch:',count)
            
            if count == num_batches:
                break

            sparse_y = helper._batch_y(y,vocabulary)

            feed_train = {
                             inputs:x,targets:sparse_y,
                             time_steps:np.array([seq_len]*actual_batch_size),
                             dropout_conv:1,dropout_fc:1,dropout_lstm:1,
                             is_training:True
                        }
            
#             print('Going for backprop and loss')
            
#             print('running optimizer')
            _,loss_val = sess.run([train,loss],feed_dict=feed_train)
#             print('ran optimizer')
            
            #loss_val = sess.run(loss,feed_dict = feed_train)
    
#             print('Came out of backprop and loss')
            
            train_loss += loss_val
            
            count+=1
            
            
        train_loss /= num_batches         
              
        #Save and validate
        if (e%save_epoch)==0:
            
            valid_loss,ler = 0.0,0.0
            
            count = 0
            
            print('Validation Loop')
            
            for xv,yv in valid_generator:
                
                
                if count == num_vbatches:
                    break
                
#                 print('Valid Minibatch',count)
                
                #Validatation feed...
                
                
                valid_size = xv.shape[0]
                sparse_targets = helper._batch_y(yv,vocabulary)

                feed_valid = {
                             inputs:xv,targets:sparse_targets,
                             time_steps:np.array([seq_len]*valid_size),
                             dropout_conv:1,dropout_fc:1,dropout_lstm:1,
                             is_training:False
                    }

            
                v_loss_val, v_ler = sess.run([loss,label_error_rate],feed_dict = feed_valid)
                
                valid_loss += v_loss_val
                ler += v_ler
                
                count+=1
                
            valid_loss /= num_vbatches
            ler /= num_vbatches
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            print("Epoch: {}, train_loss:{:.2f}, valid_loss:{:.2f}, ler:{:.2f} in {:.2f} sec.".format(e,train_loss,valid_loss,ler,time_taken)) 
            
            #Save the model
            saver.save(sess,mount_point+'saved_models/cnn_lstm_fc_'+str(e))

            with open('progress.csv','a') as f:
                f.write("Epoch: {}, train_loss:{:.2f}, valid_loss:{:.2f}, ler:{:.2f}, {:.2f} sec.\n".format(e,train_loss,valid_loss,ler,time_taken))

