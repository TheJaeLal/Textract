
# coding: utf-8

# In[1]:


from helper import create_batches
import tensorflow as tf
import numpy as np
import shelve
import joblib
import random
import time
import sys

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
alpha = 0.0001
epochs = 200
batch_size = 32

#Conv_net Params
filter_size = 5
#Number of filters in each convolution layer
num_conv1,num_conv2,num_conv3 = (16, 32, 64)

#LSTM Params
rnn_hidden_units = 128
rnn_layers = 2

#FC_Params
#hidden layer should be two times vocabulary intuitively
fc_input_units,fc_hidden_units,fc_output_units = (2*rnn_hidden_units, 2*vocab_size, vocab_size)


# In[3]:


# with tf.device('/gpu:0'):

#Weights Initializer
fc_initializer = tf.contrib.layers.xavier_initializer()
conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()

#Weights for convolution layer
# -> filter_size = 5 so filter = (5 x 5)
#-> input_channels or (channels_in_image) = 1 
#-> output_channels or (num_of_filters) = num_conv1

wconv1_shape = [filter_size,filter_size,1,num_conv1]
wconv2_shape = [filter_size,filter_size,num_conv1,num_conv2]
wconv3_shape = [filter_size,filter_size,num_conv2,num_conv3]

wfc1_shape = [fc_input_units, fc_hidden_units]
wfc2_shape = [fc_hidden_units, fc_output_units]


#Biases for conv_layer (single value, thus shape is empty tensor [])
bconv_shape = []

#Biases for fc layer
bfc1_shape = [fc_hidden_units]
bfc2_shape = [fc_output_units]

#Initialize weights 
wconv1 = tf.Variable(conv_initializer(wconv1_shape))
wconv2 = tf.Variable(conv_initializer(wconv2_shape))
wconv3 = tf.Variable(conv_initializer(wconv3_shape))

wfc1 = tf.Variable(fc_initializer(wfc1_shape))
wfc2 = tf.Variable(fc_initializer(wfc2_shape))


#Intialize biases
bconv1 = tf.Variable(tf.zeros(bconv_shape))
bconv2 = tf.Variable(tf.zeros(bconv_shape))
bconv3 = tf.Variable(tf.zeros(bconv_shape))

bfc1 = tf.Variable(tf.zeros(bfc1_shape))
bfc2 = tf.Variable(tf.zeros(bfc2_shape))


#Model
#----------------------------------------------------------------------------#

#Input Image
inputs = tf.placeholder(tf.float32,shape=[None,img_height,img_width])

X = tf.reshape(inputs,(-1,img_height,img_width,1))

#-------------------Convolution-----------------------#
#1st Convolutional Layer
conv1 = tf.nn.relu(tf.nn.conv2d(input=X,filter=wconv1,padding='SAME',strides=[1,1,1,1]) + bconv1)

#1st Pooling layer
pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#2nd Convolutional Layer
conv2 = tf.nn.relu(tf.nn.conv2d(input=pool1,filter=wconv2,padding='SAME',strides=[1,1,1,1]) + bconv2)

#2nd Pooling Layer
pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#3rd Convolutional Layer
conv3 = tf.nn.relu(tf.nn.conv2d(input=pool2,filter=wconv3,padding='SAME',strides=[1,1,1,1]) + bconv3)

#3rd Pooling Layer
pool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

conv_out_height, conv_out_width = (int(img_height/(2**3)),int(img_width/(2**3)))

#----------------LSTM--------------------------#
#Treat a single pixel from each filter or feature map as an individual feature
#So number of features  = num_conv3 filters or feature maps
#length_of_sequence = width * height of the output from conv3 

lstm_inputs = tf.reshape(pool3,(-1,conv_out_height*conv_out_width,num_conv3))

#Number of time_steps to unroll for..
seq_len = conv_out_height * conv_out_width

targets = tf.sparse_placeholder(tf.int32,name='targets')

time_steps = np.array([seq_len]*batch_size)

# RNN Cells forward
cells_fw = [tf.contrib.rnn.LSTMCell(rnn_hidden_units,initializer=fc_initializer) for _ in range(rnn_layers)]

# RNN Cells backward
cells_bw = [tf.contrib.rnn.LSTMCell(rnn_hidden_units,initializer=fc_initializer) for _ in range(rnn_layers)]

# (outputs_fw,outputs_bw),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,lstm_inputs,parallel_iterations=128,dtype=tf.float32)
outputs,_,_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,cells_bw,inputs = lstm_inputs,dtype=tf.float32)

# #Concatenate the output from both cells (forward and backward)
# blstm_outputs = tf.concat([outputs_fw,outputs_bw], 2)
                            
#convert them to time major
lstm_outputs = tf.transpose(outputs,[1,0,2])

#flatten out all except the last dimension
fc_inputs  = tf.reshape(lstm_outputs,[-1,2*rnn_hidden_units])

#Feed into the fully connected layer
#No activation cuz, the output of this layer is feeded into CTC Layer as logits

fc_outputs_1 = tf.matmul(fc_inputs,wfc1) + bfc1

fc_outputs_2 = tf.matmul(fc_outputs_1,wfc2) + bfc2

#Reshape back to time major giving logits
logits = tf.reshape(fc_outputs_2,[seq_len,-1,vocab_size])

#Calculate loss
loss = tf.nn.ctc_loss(targets, logits, time_steps,preprocess_collapse_repeated=True)
cost = tf.reduce_mean(loss)

#Optimize
optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
train = optimizer.minimize(loss)

# CTC decoder.

#decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, time_steps)

label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                   targets))


# ## Save my MoDel

# In[4]:


losses = []
saver = tf.train.Saver()

training_list = list_of_images[:256]
random.seed(100)
random.shuffle(training_list)


# In[5]:


batches_x,batches_y = create_batches(batch_size,training_list,image_arrays,image_labels,vocabulary)
print(len(batches_x),len(batches_y))


# In[6]:


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     feed = {inputs:batches_x[0].transpose([2,0,1]),targets:batches_y[0]}
#     true_fc_outputs = sess.run(fc_outputs,feed_dict = feed)
    


# In[7]:


# true_fc_outputs.shape


# In[ ]:


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(epochs): 
        start_time = time.time()
        total_cost,total_ler = 0.0,0.0
        
        #Iterate through all images in a single epoch...
        for b in range(len(batches_x)):
            
            #Before feeding x reshape it as (batch_size,width,height)
            feed = {inputs:batches_x[b].transpose([2,0,1]),targets:batches_y[b]}

            sess.run(train,feed_dict=feed)
            
        if e % 1 == 0:
            cost_val,ler_val,d = sess.run([cost,label_error_rate,decoded], feed_dict=feed)
            total_cost+=cost_val
            total_ler+=ler_val
            
            losses.append(total_cost)
#             if e % 1 == 0:
#                 outputs.append(d)
                
        if e%10==0:
            saver.save(sess,'../model/Lines_RNN_'+str(e))

        end_time = time.time()       
        time_taken = end_time - start_time

        with open('output.txt','a') as f:
            f.write("Epoch {}: cost = {} ler = {:.2f} - Time taken:{:.2f} sec\n".format(e,total_cost,total_ler,time_taken))

# plt.plot(list(range(len(losses))),losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')


# ## Predict using Model

# In[ ]:


# with tf.Session() as sess:
#     # Load the graph
#     saver = tf.train.import_meta_graph('../model/Lines_RNN_50.meta')
#     # Restore the weights and biases
#     saver.restore(sess, '../model/Lines_RNN_50')

#     inputs = sess.graph.get_tensor_by_name('Placeholder:0')
#     target_indices = 
#     target_values = 
#     target_shape = 
#     for _ in range(30): 
#         y_pred = sess.run('rnn/transpose:0', feed_dict={X: X_new})
#         #print(y_pred)
#         #print(y_pred.shape)
#         content += interpret(y_pred,vocabulary)
#         X_new = y_pred
        
# print(content)


# In[ ]:


# content = []
# for k in range(0,5):
#     val = outputs[k][0]
#     content.append(''.join([vocabulary[x] for x in np.asarray(val[1])]))
# print("\n".join(content))

