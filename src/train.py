import tensorflow as tf
import numpy as np

from math import ceil
import time
import os

from model import ANN_Model
from train_config import mount_point,vocabulary,batch_size,valid_batch_size,n_epochs,resume_epoch,save_epoch,summary_epoch,dropout
import datagen
from Arch import CNN
import layers
import helper

#Initializer the model/graph
model = ANN_Model()

graph = model[0]
dropout_lstm = model[1]
dropout_fc = model[2]
inputs = model[3]
time_steps = model[4]
targets = model[5]
loss = model[6]
train = model[7]
decoded = model[8]
label_error_rate = model[9]
seq_len = model[10]
is_training = model[11]
conv_dropout = model[12]
gradients = model[13]
interim_dropout = model[14]


train_generator, valid_generator = datagen.get_generators()

num_batches = int(ceil(datagen.num_train/batch_size))
num_vbatches = int(ceil(datagen.num_valid/valid_batch_size))

with tf.Session(graph = graph) as sess:
        
    #Get the saver object (to save and restore model params)
    saver = tf.train.Saver(max_to_keep=None)
    
    sess.run(tf.global_variables_initializer())
    checkpoint = False
    timer  = 0
                
    #Tensorboard Summary
    
    if summary_epoch:
        merged_summary = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('../visualize', sess.graph)
        file_writer.add_graph(sess.graph)
    
    #Resume training from resume_epoch
    if resume_epoch != 0:
        saver.restore(sess, os.path.join(mount_point,'saved_models','cnn_lstm_fc_'+str(resume_epoch)))
    
    #Epoch Loop
    for e in range(resume_epoch,n_epochs):
        start_time = time.time()
        
        train_loss = 0.0    
        count = 0
        
        #Mini Batch loop
        for x,y in train_generator:
            
            #Keep only the 1st channel...
            x = x[:,:,:,0]
            x = np.expand_dims(x,axis=-1)
            
#             plt.imshow(x[0].reshape(x[0].shape[:2]),cmap='gray')
#             print(y[0])
#             print(y.shape)
            
            #y,widths = np.hsplit(y,2)
#             print(y.shape)
#             print(y[0])
            
            #print("shape of y",y.shape)
        
            #widths = np.squeeze(widths,axis=1)
            #y = np.squeeze(y,axis=-1)
            
            #widths = [layers.calc_out_dims(CNN,0,int(width))[1] for width in widths]
            #widths = np.array(widths)
            
            actual_batch_size = x.shape[0]
            
            if count == num_batches:
                break
            
            #Convert targets to sparse tensor (required for CTCLoss function)
            sparse_y = helper._batch_y(y,vocabulary)

            feed_train = {
                             inputs:x,targets:sparse_y,
                             time_steps:[seq_len]*actual_batch_size,
                             conv_dropout:dropout['conv'],dropout_fc:dropout['fc'],dropout_lstm:dropout['lstm'],
                             interim_dropout:dropout['interim_fc'],is_training:True
                        }
            
            _,loss_val = sess.run([train,loss],feed_dict=feed_train)
            
            train_loss += loss_val
            
            count+=1
            
        #Average training loss..
        train_loss /= num_batches         
              
        #Save and validate
        if (e%save_epoch)==0:
                
            valid_loss,ler = 0.0,0.0
            
            count = 0

            for xv,yv in valid_generator: 
                
                if count == num_vbatches:
                    break
                
                
                #yv,widths = np.hsplit(yv,2)

                #widths = np.squeeze(widths,axis=1)
                #yv = np.squeeze(yv,axis=1)

                #widths = [layers.calc_out_dims(CNN,0,int(width))[1] for width in widths]
                #widths = np.array(widths)

                valid_size = xv.shape[0]
                sparse_targets = helper._batch_y(yv,vocabulary)
                
                #Validatation feed...
                feed_valid = {
                             inputs:xv,targets:sparse_targets,
                             time_steps:[seq_len]*valid_size,
                             conv_dropout:[1]*len(CNN),dropout_fc:1,dropout_lstm:1,
                             interim_dropout:1,is_training:False
                    }

                v_loss_val, v_ler= sess.run([loss,label_error_rate],feed_dict = feed_valid)
                
                #Write Tensorboard summaries to file
                if (summary_epoch and e%summary_epoch == 0):
                    s = sess.run(merged_summary,feed_dict=feed_valid)
                    file_writer.add_summary(s,e)
                
                valid_loss += v_loss_val
                ler += v_ler
                
                count+=1
            
            #Average loss and ler..
            valid_loss /= num_vbatches
            ler /= num_vbatches
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            #print("Epoch: {}, train_loss:{:.2f}, valid_loss:{:.2f}, ler:{:.2f} in {:.2f} sec.".format(e,train_loss,valid_loss,ler,time_taken)) 
            
            #Save the model
            saver.save(sess,mount_point+'saved_models/cnn_lstm_fc_'+str(e))

            with open('progress.csv','a') as f:
                f.write("Epoch: {}, train_loss:{:.2f}, valid_loss:{:.2f}, ler:{:.2f}, {:.2f} sec.\n".format(e,train_loss,valid_loss,ler,time_taken))