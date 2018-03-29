
# coding: utf-8

# In[1]:


from helper import create_batches
import tensorflow as tf
import numpy as np
import shelve
import joblib
import random
import time
import math

#Cuz the file is inside 'code' directory
mount_point = "../shelved_data/"

with shelve.open(mount_point+'IAM_Data') as shelf:
    vocabulary = shelf['chars']
    list_of_images = shelf['list_of_images']
    image_labels = shelf['image_labels']
    
image_arrays = joblib.load(mount_point+'image_arrays')

#List_images ko sort karo
list_of_images.sort()

#Convert vocabulary to list
vocabulary = list(vocabulary)
#Sort so as to have the same ordering every time..
vocabulary.sort()
vocabulary.append("<Blank>")


# In[2]:


#Model parameters
img_height = 104
img_width = 688
vocab_size = len(vocabulary)

#Common Hyper Parameters
alpha = 0.0005
epochs = 400
batch_size = 250 #Should be proportional to the number of Images

#LSTM Params
rnn_hidden_units = 200
# rnn_layers = 1

#FC_Params
#hidden layer should be two times vocabulary intuitively
fc_input_units,fc_hidden_units,fc_output_units = (2*rnn_hidden_units, 2*vocab_size, vocab_size)

conv_out_height, conv_out_width = (int(math.ceil(img_height/(2**3 * 3))),int(math.ceil(img_width/(2**3 * 3))))

#Number of time_steps to unroll for..
seq_len = conv_out_height * conv_out_width


# ## Save my MoDel

# In[3]:


#Take first 12000 images...
training_list = list_of_images[:12000]


# In[4]:


batches_x,batches_y = create_batches(batch_size,training_list,image_arrays,image_labels,vocabulary)
print(len(batches_x),len(batches_y))


# ## Predict using Model

# In[5]:


resume_epoch = 49


# In[6]:


with tf.Session() as sess:
    # Load the graph
    saver = tf.train.import_meta_graph('../model/Lines_RNN_'+str(resume_epoch)+'.meta')

    # Restore the weights and biases
    saver.restore(sess, '../model/Lines_RNN_'+str(resume_epoch))

    inputs = sess.graph.get_tensor_by_name('Placeholder:0')
    target_indices = sess.graph.get_tensor_by_name('targets/indices:0')
    target_values = sess.graph.get_tensor_by_name('targets/values:0')
    target_shape = sess.graph.get_tensor_by_name('targets/shape:0')

    time_steps = sess.graph.get_tensor_by_name('Placeholder_1:0')
    dropout_lstm = sess.graph.get_tensor_by_name('Placeholder_2:0')
    dropout_fc = sess.graph.get_tensor_by_name('Placeholder_3:0')
    
    train = sess.graph.get_operation_by_name('RMSProp')
    cost = sess.graph.get_tensor_by_name('Mean:0')
    label_error_rate = sess.graph.get_tensor_by_name('Mean_1:0')

    for e in range(resume_epoch,epochs): 
        start_time = time.time()
        total_cost,total_ler = 0.0,0.0

        #Iterate through all images in a single epoch...
        for b in range(len(batches_x)):
            
            feed = {
                    inputs:batches_x[b].transpose([2,0,1]),
                    target_indices:batches_y[b][0],target_values:batches_y[b][1],target_shape:batches_y[b][2],
                    time_steps:np.array([seq_len]*batch_size),
                    dropout_fc:np.array(0.7),dropout_lstm:np.array(0.8)
                   }
            
            sess.run(train,feed_dict=feed)

        if e % 1 == 0:
            cost_val,ler_val = sess.run([cost,label_error_rate], feed_dict=feed)
            total_cost+=cost_val
            total_ler+=ler_val
            
#             if e % 1 == 0:
#                 outputs.append(d)
                
        if e%10==9:
            saver.save(sess,'../model/Lines_RNN_'+str(e))

        end_time = time.time()       
        time_taken = end_time - start_time

        print(total_cost)
        with open('progress.csv','a') as f:
            f.write("{},{:.6f},{:.2f},{:.2f}\n".format(e,total_cost,total_ler,time_taken))

