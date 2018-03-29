
# coding: utf-8

# In[1]:


from helper import create_batches
import tensorflow as tf
import numpy as np
import shelve
import joblib
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
epochs = 650
batch_size = 250 #Should be proportional to the number of Images

#LSTM Params
rnn_hidden_units = 200

conv_out_height, conv_out_width = (int(math.ceil(img_height/(2**3 * 3))),int(math.ceil(img_width/(2**3 * 3))))

#Number of time_steps to unroll for..
seq_len = conv_out_height * conv_out_width


# ## Save my MoDel

# In[3]:


#Take first 12000 images...
training_list = list_of_images[:12000]
testing_list = list_of_images[12000:13000]


# In[4]:


train_batches_x,train_batches_y = create_batches(batch_size,training_list,image_arrays,image_labels,vocabulary)
test_batches_x,test_batches_y = create_batches(len(testing_list),testing_list,image_arrays,image_labels,vocabulary)


# ## Predict using Model

# In[5]:


resume_epoch = 15


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

    is_test = sess.graph.get_tensor_by_name('Placeholder_1:0')
    iteration = sess.graph.get_tensor_by_name('Placeholder_2:0')
    time_steps = sess.graph.get_tensor_by_name('Placeholder_3:0')
    dropout_lstm = sess.graph.get_tensor_by_name('Placeholder_4:0')
    dropout_fc = sess.graph.get_tensor_by_name('Placeholder_5:0')
    
        
    train = sess.graph.get_operation_by_name('RMSProp')
    update_ema = sess.graph.get_operation_by_name('group_deps')
    cost = sess.graph.get_tensor_by_name('Mean:0')
    label_error_rate = sess.graph.get_tensor_by_name('Mean_1:0')

    #After every 5 epochs
    checkpoint = False
    
    timer = 0
    
    for e in range(resume_epoch,epochs): 

        start_time = time.time()
    
        if (e%5)==0:
            checkpoint = True

        #Iterate through all images in a single epoch...
        for b in range(len(train_batches_x)):
            
            feed_train = {
                    inputs:train_batches_x[b].transpose([2,0,1]),
                    target_indices:train_batches_y[b][0],target_values:train_batches_y[b][1],target_shape:train_batches_y[b][2],
                    time_steps:np.array([seq_len]*batch_size),
                    is_test:False,
                    dropout_fc:np.array(0.7),dropout_lstm:np.array(0.8)
                   }

            feed_uma = {
                    inputs:train_batches_x[b].transpose([2,0,1]),
                    target_indices:train_batches_y[b][0],target_values:train_batches_y[b][1],target_shape:train_batches_y[b][2],
                    time_steps:np.array([seq_len]*batch_size),
                    is_test:False,iteration:len(train_batches_x)*e + b,
                    dropout_fc:np.array(1.0),dropout_lstm:np.array(1.0)
                   }

        
            sess.run(train,feed_dict=feed_train)
            sess.run(update_ema,feed_dict=feed_uma)
            

        if checkpoint:
            last_cost = sess.run(cost,feed_dict=feed_train)

        #After iterating through all batches..
        test_batch_size = len(testing_list)
      
        feed_test = {
            inputs:test_batches_x[0].transpose([2,0,1]),
            target_indices:test_batches_y[0][0],target_values:test_batches_y[0][1],target_shape:test_batches_y[0][2],
            time_steps:np.array([seq_len]*test_batch_size),
            is_test:True,
            dropout_fc:np.array(1.0),dropout_lstm:np.array(1.0)
           }
            

        #Evaluate the model, and store every 5 epochs...
        if checkpoint:

            #Accuracy on test_data
            ler_val = sess.run(label_error_rate,feed_dict=feed_test)                

            end_time = time.time()       
            time_taken = end_time - start_time
            timer += time_taken
                        
#             print("{},{:.6f},{:.2f},{:.2f}\n".format(e,last_cost,ler_val,timer))
            with open('progress.csv','a') as f:
                f.write("{},{:.6f},{:.2f},{:.2f}\n".format(e,last_cost,ler_val,timer))

            #Save the model
            saver.save(sess,'../model/Lines_RNN_'+str(e))

            checkpoint = False
            timer = 0
        
        else:
            end_time = time.time()       
            time_taken = end_time - start_time
            timer += time_taken

