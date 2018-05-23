import tensorflow as tf
import numpy as np
import layers
from config import vocab_size, img_width,img_height, alpha,cell_clip,first_decay_steps,t_mul,m_mul,momentum,decay
from Arch import CNN, BRNN, FC
import sys

#Global Debugging flag..
debug = True

# with tf.device('/gpu:0'):

#Model
#----------------------------------------------------------------------------#
def model():
    
    graph = tf.Graph()
    with graph.as_default():
    
#         wconv,bconv,wfc,bfc = layers.init_weights(CNN,FC)
        
        #Placeholder required for batch_norm
        
        is_training = tf.placeholder(tf.bool)
        conv_dropout = tf.placeholder(tf.float32,shape=len(CNN))
    
        learning_rate = tf.placeholder(tf.float32)
        dropout_lstm = tf.placeholder(tf.float32)
        dropout_fc = tf.placeholder(tf.float32)

#	learning_rate = tf.placehodler(tf.float32)

        #Input 'Image'
        inputs = tf.placeholder(tf.float32,shape=[None,img_height,img_width,1])
        

        #tf.summary.image('images',inputs)
        
        X = inputs

        #-------------------Convolution-----------------------#

#         conv = [None] * len(CNN)

        #Create your CNN
        for i in range(len(CNN)):
            
            kernel_size,strides,num_filters = CNN[i]['conv']
            activation = CNN[i]['activate']
            
#             X = tf.Print(X,[X],'Before_conv_'+str(i+1)+':')
            
            conv = layers.conv(X,num_filters,kernel_size,strides,activation,is_training,conv_dropout[i])
            
            if CNN[i]['pool']:
                conv = layers.max_pool(conv,CNN[i]['pool'])
            
#             conv = tf.Print(conv,[conv],'Conv_layer_'+str(i+1)+':')
                
            #Input to next layer is output of previous layer...
            X = conv
        
        #Calculate height and width of output from CNN
        conv_out_height,conv_out_width = layers.calc_out_dims(CNN,img_height,img_width)
        if debug:
            print('Convolution_Output_size:({},{})'.format(conv_out_height,conv_out_width))
        
            
        #----------------LSTM--------------------------#
        #Treat a single pixel from each filter or feature map as an individual feature
        #So number of features  = num_conv4 filters or feature maps
        #length_of_sequence = width * height of the output from conv3 

        filters_in_last_conv = CNN[-1]['conv'][2]
        
        #Number of time_steps to unroll for..
        seq_len = conv_out_width
        num_features = conv_out_height*filters_in_last_conv
        
        lstm_inputs = tf.reshape(conv,(-1,seq_len,num_features))
        
        #So that we can use different batch size during testing...
        time_steps = tf.placeholder(tf.int32,shape = [None])
                
        targets = tf.sparse_placeholder(tf.int32,name='targets')

        lstm_initializer = tf.contrib.layers.xavier_initializer()
        
        cells_fw = layers.lstm(BRNN['layers'],cell_clip,lstm_initializer,dropout=dropout_lstm)

        cells_bw = layers.lstm(BRNN['layers'],cell_clip,lstm_initializer,dropout=dropout_lstm)
        
#         (outputs_fw,outputs_bw),_ = tf.nn.bidirectional_dynamic_rnn(fw_layer,bw_layer,lstm_inputs,dtype=tf.float32)
        
#         outputs_fw = tf.Print(outputs_fw,[outputs_fw],'LSTM_Forward:')
#         outputs_bw = tf.Print(outputs_bw,[outputs_bw],'LSTM_Backward:')

#-------> #By Default batch_major
        #Try time_major (later to see performance diff...)
        
        blstm_outputs,_,_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,cells_bw,inputs = lstm_inputs,dtype=tf.float32)

        if debug:
            print('LSTM_Output_size:',blstm_outputs.shape)

#----->  #Stacked blstm does automatic depth concatenation.....
        
#         Concatenate the output from both cells (forward and backward)
#         blstm_outputs = tf.concat([outputs_fw,outputs_bw], 2)

        #flatten out all except the last dimension
        fc_inputs  = tf.reshape(blstm_outputs,[-1,2*BRNN['layers'][-1]])
#         fc_inputs = tf.layers.batch_normalization(fc_inputs,scale=True,training=is_training,fused=False)
        
#         fc_inputs = tf.Print(fc_inputs,[fc_inputs],'FC_Inputs:')

        
        #Feed into the fully connected layer
        #No activation cuz, the output of this layer is feeded into CTC Layer as logits
        for i in range(len(FC)):
            fc_out = layers.fc(fc_inputs,FC[i]['units'],FC[i]['activate'],is_training,dropout_fc)
            
#             fc_out = tf.Print(fc_out,[fc_out],'FC_Layer_'+str(i)+':')
            
            #Input to next layer is output of previous layer..
            fc_inputs = fc_out

        #Reshape back to batch_size, seq_len,vocab_size
        logits = tf.reshape(fc_out,[-1,seq_len,vocab_size])

#         logits = tf.Print(logits,[logits],'logits:')
        
        #convert them to time major
        logits = tf.transpose(logits,[1,0,2])

        #Calculate loss
        loss = tf.reduce_mean(tf.nn.ctc_loss(targets, logits, time_steps,preprocess_collapse_repeated=True))
    
        #L2 regularization stuff..
#         all_vars = tf.trainable_variables(scope=None)
#         l2 = decay * tf.add_n( [tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() 
#                                     if not ("noreg" in tf_var.name or "bias" in tf_var.name)])
        
#         loss += l2
        
        
        #tf.summary.scalar('loss',loss)
        
        loss = tf.Print(loss,[loss],'loss:')
        
        
        #Cyclic learning rate schedule
        #global_step = tf.Variable(0, trainable=False)
               
        #lr_decayed = tf.train.cosine_decay_restarts(alpha, global_step,first_decay_steps,
        #                                   t_mul=t_mul,m_mul=m_mul)
        
#        #Optimize
#         optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_decayed)
#         train = optimizer.minimize(loss,global_step = global_step)
        

#         train = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(loss)
          
        #Trying Adam Optimizer....
    # control dependencies for batch norm ops
#         with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):



#         for var in all_vars:
#             print(var)
#        gpus = layers.get_available_gpus()
        
#        if len(gpus)>1:
#            train = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(loss,colocate_gradients_with_ops=True)

#        else:
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

#         train = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
        
        # CTC decoder.
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, time_steps)
        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))
        
        #tf.summary.scalar('label_error_rate',label_error_rate)
        
        return graph,dropout_lstm,dropout_fc,inputs,time_steps,targets,loss,train,decoded,label_error_rate,seq_len,is_training,conv_dropout,learning_rate