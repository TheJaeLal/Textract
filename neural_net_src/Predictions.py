import tensorflow as tf
import numpy as np
import shelve
import joblib
from math import ceil
import time

import neural_net_src.layers as layers
import neural_net_src.model as model
from neural_net_src.Augment import valid_datagen
import neural_net_src.helper as helper
from neural_net_src.test_config import vocabulary,test_batch_size,mount_point,prediction_set, model_dir, model_prefix
from neural_net_src.Arch import CNN


#Generating Labels and image names
with shelve.open('Metadata','c') as shelf:
    valid_label = shelf[prediction_set+'_label']

valid_array = joblib.load(os.path.join('data',prediction_set+'_array'))

#Importing model parameters
model_params = model.model()

graph = model_params[0]
dropout_lstm = model_params[1]
dropout_fc = model_params[2]
inputs = model_params[3]
time_steps = model_params[4]
targets = model_params[5]
loss = model_params[6]
train = model_params[7]
decoded = model_params[8]
label_error_rate = model_params[9]
seq_len = model_params[10]
is_training = model_params[11]
conv_dropout = model_params[12]
gradients = model_params[13]
interim_dropout = model_params[14]

#Generating images
valid_generator = valid_datagen.flow(valid_array,valid_label,valid_batch_size)

num_valid_samples = valid_array.shape[0]

num_vbatches = 1

#Inputs for images, outputs for predictions, targets for labels
infer_inputs = []
infer_outputs = []
infer_targets = []


with tf.Session(graph = graph) as sess:
        
    saver = tf.train.Saver(max_to_keep=None)
    
    sess.run(tf.global_variables_initializer())
    timer  = 0

    saver.restore(sess, os.path.join(model_dir, model_prefix+str(resume_epoch)))

    start_time = time.time()

    count = 0

    for x,y in valid_generator:
       
        infer_inputs.append(x)
        infer_targets.append(y)
        
        actual_batch_size = x.shape[0]

        if count == num_vbatches:
            break

        y,widths = np.hsplit(y,2)

        widths = np.squeeze(widths,axis=1)
        y = np.squeeze(y,axis=1)

        widths = [layers.calc_out_dims(CNN,0,int(width))[1] for width in widths]
        widths = np.array(widths)
 
        sparse_y = helper._batch_y(y,vocabulary)
            
        feed = {
                         inputs:x,targets:sparse_y,
                         time_steps:widths,conv_dropout:[1]*len(CNN),
                         dropout_fc:1,dropout_lstm:1,interim_dropout:1,
                         is_training:False
                }
        
        loss_val,d = sess.run([loss,decoded],feed_dict=feed)
        infer_outputs.append(d)
        count+=1

#Print loss value
print(loss_val)

#Predicted string
content = "".join([vocabulary[char] for char in infer_outputs[0][0][1]])
print(content)

#Validating with input image
original_img = infer_inputs[0][0]
original_img = original_img.reshape(original_img.shape[:2])

#original_img.shape
plt.imsave('test_input.jpg',original_img,cmap='gray',format='jpg')

original_label = infer_targets[0][0]
print(original_label)