import tensorflow as tf
import numpy as np
import shelve
import joblib
from math import ceil
import time
import model
from Augment import train_datagen, valid_datagen
import helper
from train_config import vocabulary,batch_size,valid_batch_size,n_epochs,resume_epoch,save_epoch,momentum,summary_epoch,dropout
from Arch import CNN
import layers

#Unused Import...
# import matplotlib.pyplot as plt

mount_point = '../'

with shelve.open(mount_point+'IAM_Data','c') as shelf:
    train_label = shelf['train_label']
    valid_label = shelf['valid_label']

    #For testing overfitting..
#     valid_label = shelf['train_label'][:1024]
    
#     If more data needed use zombie data
    zombie_label = shelf['zombie_label'][:1024]
    
    #For finding accuracy on test data
    #test_label = shelf['test_label']


train_array = joblib.load(mount_point+'data/train_array')
valid_array = joblib.load(mount_point+'data/valid_array')

#For testing overfitting.. 
# valid_array = joblib.load(mount_point+'data/train_array')[:1024]

#If additional data is needed...
zombie_array = joblib.load(mount_point+'data/zombie_array')[:1024]

#For finding accuracy on test data
#test_array = joblib.load(mount_point+'data/test_array')

##*****Increase the training dataset...*****
##Add the zombie array to the train_array and do the same for labels...
train_array = np.concatenate((train_array,zombie_array))
train_label = np.concatenate((train_label,zombie_label))

#Initializer the model/graph
graph,dropout_lstm,dropout_fc,inputs,time_steps,targets,loss,train,decoded,label_error_rate,seq_len,is_training,conv_dropout,gradients,interim_dropout = model.model()

##If you want to see augmented Images...
# train_generator = train_datagen.flow(train_array,train_label,batch_size,save_to_dir=mount_point+'Augmented', save_prefix='train')

train_generator = train_datagen.flow(train_array,train_label,batch_size)
valid_generator = valid_datagen.flow(valid_array,valid_label,valid_batch_size)
#test_generator = valid_datagen.flow(test_array,test_label,valid_batch_size)

num_training_samples = train_array.shape[0]
num_valid_samples = valid_array.shape[0]
#num_test_samples = test_array.shape[0]

num_batches = int(ceil(num_training_samples/batch_size))
num_vbatches = int(ceil(num_valid_samples/valid_batch_size))
#num_vbatches = int(ceil(num_test_samples/valid_batch_size))

with tf.Session(graph = graph) as sess:
        
    saver = tf.train.Saver(max_to_keep=None)
    
    sess.run(tf.global_variables_initializer())
    checkpoint = False
    timer  = 0
                
    merged_summary = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter('../visualize', sess.graph)
    file_writer.add_graph(sess.graph)
    
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
#             print(y.shape)
            y,widths = np.hsplit(y,2)
#             print(y.shape)
#             print(y[0])
            
            #widths = np.squeeze(widths,axis=1)
            y = np.squeeze(y,axis=1)
            
            #widths = [layers.calc_out_dims(CNN,0,int(width))[1] for width in widths]
            #widths = np.array(widths)
            
            actual_batch_size = x.shape[0]
            
#             print('Train Minibatch:',count)
            
            if count == num_batches:
                break

            sparse_y = helper._batch_y(y,vocabulary)

            feed_train = {
                             inputs:x,targets:sparse_y,
                             time_steps:[seq_len]*actual_batch_size,
                             conv_dropout:dropout['conv'],dropout_fc:dropout['fc'],dropout_lstm:dropout['lstm'],
                             interim_dropout:dropout['interim_fc'],is_training:True
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
            
#             print('Validation Loop')
            
#Change this to valid, once done with testing...
            for xv,yv in valid_generator: 
                
                if count == num_vbatches:
                    break
                
#                 print('Valid Minibatch',count)
                
                #Validatation feed...

                yv,widths = np.hsplit(yv,2)

                #widths = np.squeeze(widths,axis=1)
                yv = np.squeeze(yv,axis=1)

                #widths = [layers.calc_out_dims(CNN,0,int(width))[1] for width in widths]
                #widths = np.array(widths)

                valid_size = xv.shape[0]
                sparse_targets = helper._batch_y(yv,vocabulary)
                
                feed_valid = {
                             inputs:xv,targets:sparse_targets,
                             time_steps:[seq_len]*valid_size,
                             conv_dropout:[1]*len(CNN),dropout_fc:1,dropout_lstm:1,
                             interim_dropout:1,is_training:False
                    }

                v_loss_val, v_ler= sess.run([loss,label_error_rate],feed_dict = feed_valid)
        
                
                if (e%summary_epoch == 0):
                    s = sess.run(merged_summary,feed_dict=feed_valid)
                    file_writer.add_summary(s,e)
                
                valid_loss += v_loss_val
                ler += v_ler
                
                count+=1
                
            valid_loss /= num_vbatches
            ler /= num_vbatches
            
            end_time = time.time()
            time_taken = end_time - start_time
            
#             print("Epoch: {}, train_loss:{:.2f}, valid_loss:{:.2f}, ler:{:.2f} in {:.2f} sec.".format(e,train_loss,valid_loss,ler,time_taken)) 
            
            #Save the model
            saver.save(sess,mount_point+'saved_models/cnn_lstm_fc_'+str(e))

            with open('progress.csv','a') as f:
                f.write("Epoch: {}, train_loss:{:.2f}, valid_loss:{:.2f}, ler:{:.2f}, {:.2f} sec.\n".format(e,train_loss,valid_loss,ler,time_taken))
