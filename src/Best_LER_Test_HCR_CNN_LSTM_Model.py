import tensorflow as tf
import numpy as np
import shelve
import joblib
from math import ceil
import time
import model
from Augment import valid_datagen
import helper
from test_config import vocabulary,test_batch_size,resume_epoch,model_dir
from Arch import CNN
import layers
import matplotlib.pyplot as plt

# ## Load Test Labels and Image-Array

mount_point = '../'

with shelve.open(mount_point+'IAM_Data','c') as shelf:
    test_label = shelf['test_label']
    
test_array = joblib.load(mount_point+'data/test_array')

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
        saver.restore(sess, mount_point+model_dir+'/cnn_lstm_fc_'+str(resume_epoch))

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


# ## Predict using Model

# In[ ]:


# resume_epoch = 1545


# In[ ]:


# outputs = []


# In[ ]:


# with tf.Session() as sess:
#     # Load the graph
#     saver = tf.train.import_meta_graph('../model/200_5_Lines_RNN_'+str(resume_epoch)+'.meta')
#     # Restore the weights and biases
#     saver.restore(sess, '../model/200_5_Lines_RNN_'+str(resume_epoch))

#     #Extract the placeholders
#     inputs = sess.graph.get_tensor_by_name('Placeholder:0')
#     target_indices = sess.graph.get_tensor_by_name('targets/indices:0')
#     target_values = sess.graph.get_tensor_by_name('targets/values:0')
#     target_shape = sess.graph.get_tensor_by_name('targets/shape:0')
    
#     time_steps = sess.graph.get_tensor_by_name('Placeholder_1:0')
#     dropout_lstm = sess.graph.get_tensor_by_name('Placeholder_2:0')
#     dropout_fc = sess.graph.get_tensor_by_name('Placeholder_3:0')
    
#     decoded = sess.graph.get_tensor_by_name('CTCGreedyDecoder:1')
#     cost = sess.graph.get_tensor_by_name('Mean:0')
#     label_error_rate = sess.graph.get_tensor_by_name('Mean_1:0')

#     start_time = time.time()
    
    
#     for b in range(len(batches_x)):
#         feed = {inputs:batches_x[b].transpose([2,0,1]),target_indices:batches_y[b][0],target_values:batches_y[b][1],target_shape:batches_y[b][2],
#                 time_steps:np.array([seq_len]*batch_size),dropout_lstm:1.0, dropout_fc:1.0,
#                }

#         cost_val,ler_val,d = sess.run([cost,label_error_rate,decoded], feed_dict=feed)

#         outputs.append(d)

#         end_time = time.time()   
    
#         time_taken = end_time - start_time

# #         print("{:.6f},{:.2f},{:.2f}\n".format(cost_val,ler_val,time_taken))


# In[ ]:


# len(outputs)


# In[ ]:


# plt.imsave('1.png',image_arrays[training_list[0]],cmap='gray',format='png')
# plt.imsave('2.png',image_arrays[training_list[1]],cmap='gray',format='png')
# plt.imsave('3.png',image_arrays[training_list[2]],cmap='gray',format='png')
# # plt.imsave('4.png',image_arrays[training_list[3]],cmap='gray',format='png')
# # plt.imsave('5.png',image_arrays[training_list[4]],cmap='gray',format='png')


# In[ ]:


# content = "".join([vocabulary[char] for char in outputs[0]])


# In[ ]:


# content


# In[ ]:


# target = "".join(image_labels[training_list[i]] for i in range(batch_size))


# In[ ]:


# target


# In[ ]:


# print(image_labels[training_list[4]])


# In[ ]:


# training_list[0]


# In[ ]:


# plt.imshow(batches_x[0].transpose([2,0,1])[0],cmap='gray')


# In[ ]:


# output = str(list(map(dct.get, list(prob_d.values))))


# In[ ]:


# #Evaluate the Output
# content = []
# for k in range(len(out1)):
#     content.append(''.join([vocabulary[x] for x in out1[k]]))
#     print("\n".join(content))


# In[ ]:


# vocabulary[]


# In[ ]:


# out1[0]
# for x in out1[0]:
#     print(x)


# In[ ]:


# out1[100]


# In[ ]:


#For Outputs...
# Output of CTCGreedyDecoder
# [<tf.Tensor 'CTCGreedyDecoder:0' shape=(?, 2) dtype=int64>,
#  <tf.Tensor 'CTCGreedyDecoder:1' shape=(?,) dtype=int64>,
#  <tf.Tensor 'CTCGreedyDecoder:2' shape=(2,) dtype=int64>,
#  <tf.Tensor 'CTCGreedyDecoder:3' shape=(32, 1) dtype=float32>]

