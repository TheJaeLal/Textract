import tensorflow as tf
import numpy as np

from math import ceil
import time
import os

from neural_net_src.model import ANN_Model
from neural_net_src.test_config import model_prefix
from neural_net_src.train_config import vocabulary,batch_size,valid_batch_size,n_epochs,resume_epoch,save_epoch,summary_epoch,dropout
import neural_net_src.datagen as datagen
from neural_net_src.Arch import CNN, iterations
import lneural_net_src.layers as layers
import neural_net_src.helper as helper

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
        file_writer = tf.summary.FileWriter('visualize', sess.graph)
        file_writer.add_graph(sess.graph)
    
    #Resume training from resume_epoch
    if resume_epoch != 0:
        saver.restore(sess, os.path.join('saved_models',model_prefix+str(resume_epoch)))
    
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
            
            #break down y -> paragraph into different sentences...
            #Decision to take here...
            
            #y is a batch of text
            #y is a list of strings...
            #each string represent a single sample
            #all of the strings combined together form a batch..
            #so if you split each string/paragraph into lines, you'll have to ensure, that the 
            #Batching order is maintained...
            #So first line of para 1 with 1st line of para 2 and so on till 1st line of batch_size'th para
            
            #list of lists...
            
#             print("Shape of y = ",y.shape)
#             print("y[0] = ",y[0])
            
            line_labels = [line.split("\n") for line in y]
            
            # print("Shape of line_labels = ",len(line_labels))
            # print("line_labels[0] = ",line_labels[0])
            
                
            """
            batch_size = 2
            iterations = 5
            [
                [
                "Hello World, It's Siraj",
                
                "And today we're going to explore",
                
                "Convolutional Neural Networks",
                
                "",
                
                ""]
                
                [
                "Now Convolutional Neural Networks,",
            
                "Are state of the Art in many",
                
                "Image Recognition Tasks",
                
                "that were solved using RNNs before!",
                
                ""]
            
            ] --> 
            
            to_be_fed into helper._batch_y()
            [
                    "Hello World, It's Siraj",
                    "Now Convolutional Neural Networks"
            
            ] --> returns a sparse tensor (_,_,_)
            
            And we make 32 such sparse tensors 
            
            
            """
            #Convert targets to sparse tensor (required for CTCLoss function)
            
            #Repeat this process 32 times
            
            #1st batch contains line 1 for all 
            
            #Since no_of_sparse_tensors = 32
            sparse_targets = []
            
            new_y = [""]*len(line_labels)
            new_y = [new_y]*iterations
            
            for p in range(len(line_labels)):
                for l in range(len(line_labels[p])):
                    new_y[l][p] = line_labels[p][l]
            
            
            for k in range(iterations):
        
                #the function helper._batch_y should get a list(batch) of strings/lines...
                sparse_targets.append(helper._batch_y(new_y[k],vocabulary))

                
            feed_train = {
                             inputs:x,
                             time_steps:[seq_len]*actual_batch_size,
                             conv_dropout:dropout['conv'],dropout_fc:dropout['fc'],dropout_lstm:dropout['lstm'],
                             interim_dropout:dropout['interim_fc'],is_training:True
                        }
            
            #For accomodating the sparse tensors...
            for it in range(iterations):
                feed_train[targets[it]] = sparse_targets[it]
                
                
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
            saver.save(sess,os.path.join('saved_models',model_prefix+str(e)))

            with open('progress.csv','a') as f:
                f.write("Epoch: {}, train_loss:{:.2f}, valid_loss:{:.2f}, ler:{:.2f}, {:.2f} sec.\n".format(e,train_loss,valid_loss,ler,time_taken))