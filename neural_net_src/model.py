import tensorflow as tf
import numpy as np

import neural_net_src.layers as layers
from neural_net_src.train_config import vocab_size, img_width, img_height, alpha, cell_clip, first_decay_steps, t_mul, m_mul, momentum, decay
from neural_net_src.Arch import CNN, BRNN, FC, Interim_FC, iterations

#Global Debugging flag..
debug = False

def ANN_Model():
    graph = tf.Graph()
    with graph.as_default():

#         wconv,bconv,wfc,bfc = layers.init_weights(CNN,FC)

        #Placeholder required for batch_norm

        is_training = tf.placeholder(tf.bool)
        conv_dropout = tf.placeholder(tf.float32,shape=len(CNN))

        dropout_lstm = tf.placeholder(tf.float32)
        dropout_fc = tf.placeholder(tf.float32)
        interim_dropout = tf.placeholder(tf.float32)

#	learning_rate = tf.placehodler(tf.float32)

        #Input 'Image'
        inputs = tf.placeholder(tf.float32,shape=[None,img_height,img_width,1])
        
        #Normalize the input Images...
        inputs = inputs - 128
        inputs = inputs / 128.0
        
        
        #tf.summary.image('images',inputs)

        #Necessary for it to converge, it requires input between 0 to 1.....
        X = inputs
    
        
        #-------------------Convolution-----------------------#

#         conv = [None] * len(CNN)

        #Create your CNN
        for i in range(len(CNN)):

            kernel_size,strides,num_filters = CNN[i]['conv']
            activation = CNN[i]['activate']
            padding = CNN[i]['padding']
            use_batchnorm = CNN[i]['batch_norm']
#             X = tf.Print(X,[X],'Before_conv_'+str(i+1)+':')

            conv = layers.conv(X,num_filters,kernel_size,strides,padding,activation,is_training,conv_dropout[i],use_batchnorm)

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
        num_features = filters_in_last_conv
        
#         if len(Interim_FC)!=0:

#             interim_fc_inputs = tf.reshape(conv,(-1,num_features))

#             for i in range(len(Interim_FC)):

#                 interim_fc_out = layers.fc(interim_fc_inputs,Interim_FC[i]['units'],
#                                            Interim_FC[i]['activate'],is_training,
#                                            interim_dropout)


#                 #Input to next layer is output of previous layer..
#                 interim_fc_inputs = interim_fc_out

#             #Needed so that next layer can assume that this output is coming from conv...
#             conv = interim_fc_out
#             num_features = Interim_FC[-1]['units']

        
        #So that we can use different batch size during testing...
        time_steps = tf.placeholder(tf.int32,shape = [None])
        
        total_loss,total_ler = 0,0
        targets = [tf.sparse_placeholder(tf.int32,name='target')]*iterations
    
        for line in range(iterations):
            
            sparse_target = targets[line]
            
            lstm_inputs = conv[:,line,:,:]
            
            #lstm_inputs = tf.reshape(conv,(-1,seq_len,num_features))

            lstm_initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
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

                # CTC greedy decoder.
                decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, time_steps)

                # CTC beam search decoder
                #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,time_steps)

                label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),sparse_target))

                #tf.summary.scalar('label_error_rate',label_error_rate)

                #Calculate loss
                loss = tf.reduce_mean(tf.nn.ctc_loss(sparse_target, logits, time_steps,preprocess_collapse_repeated=False,ignore_longer_outputs_than_inputs=False))

                #L2 regularization stuff..
        #         all_vars = tf.trainable_variables(scope=None)
        #         l2 = decay * tf.add_n( [tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() 
        #                                     if not ("noreg" in tf_var.name or "bias" in tf_var.name)])

        #         loss += l2


                #tf.summary.scalar('loss',loss)

                total_loss += loss
                total_ler += label_error_rate

                total_loss = tf.Print(total_loss,[total_loss],'total_loss:')

            
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



#         for var in all_vars:
#             print(var)
#        gpus = layers.get_available_gpus()

#        if len(gpus)>1:
#            train = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(loss,colocate_gradients_with_ops=True)

#        else:

        optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha)
        gradients = optimizer.compute_gradients(total_loss)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train = optimizer.apply_gradients(gradients)

        #Log all the gradients...
        for index, grad in enumerate(gradients):
            tf.summary.histogram("{}-grad".format(gradients[index][1].name), gradients[index])

        #Log all the weights...
        all_vars = tf.trainable_variables(scope=None)
        for tf_var in tf.trainable_variables():
            tf.summary.histogram("{}-weight".format(tf_var.name),tf_var)

#         train = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
        
        params = [
                  graph,dropout_lstm,dropout_fc,
                  inputs,time_steps,targets,
                  total_loss,train,decoded,total_ler,seq_len,
                  is_training,conv_dropout,gradients,interim_dropout
            ]
        
        return params 
