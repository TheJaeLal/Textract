import tensorflow as tf
import numpy as np
import layers
from config import vocab_size, img_width,img_height, alpha
from Arch import CNN, BRNN, FC


#Global Debugging flag..
debug = True

# with tf.device('/gpu:0'):

#Model
#----------------------------------------------------------------------------#
def model():
    
    graph = tf.Graph()
    with graph.as_default():
    
        wconv,bconv,wfc,bfc = layers.init_weights(CNN,FC)
        
        dropout_conv = tf.placeholder(tf.float32,shape=[])
        dropout_lstm = tf.placeholder(tf.float32,shape=[])
        dropout_fc = tf.placeholder(tf.float32,shape=[])

        #Input 'Image'
        inputs = tf.placeholder(tf.float32,shape=[None,img_height,img_width,1])

        X = inputs

        #-------------------Convolution-----------------------#

        conv = [None] * len(CNN)

        #Create your CNN
        for i in range(len(CNN)):
            strides = CNN[i]['conv'][1]
            conv[i] = layers.conv(X,wconv[i],bconv[i],strides,CNN[i]['activate'],dropout_conv)

            if CNN[i]['pool']:
                conv[i] = layers.max_pool(conv[i],CNN[i]['pool'])

            #Input to next layer is output of previous layer...
            X = conv[i]


        #Calculate height and width of output from CNN
        conv_out_height,conv_out_width = layers.calc_out_dims(CNN,img_height,img_width)
        if debug:
            print('Convolution_Output_size:({},{})'.format(conv_out_height,conv_out_width))

        #----------------LSTM--------------------------#
        #Treat a single pixel from each filter or feature map as an individual feature
        #So number of features  = num_conv4 filters or feature maps
        #length_of_sequence = width * height of the output from conv3 

        filters_in_last_conv = CNN[-1]['conv'][2]
        lstm_inputs = tf.reshape(conv[-1],(-1,conv_out_height*conv_out_width,filters_in_last_conv))

        #Number of time_steps to unroll for..
        seq_len = conv_out_height * conv_out_width

        #So that we can use different batch size during testing...
        time_steps = tf.placeholder(tf.int32,shape = [None])
        targets = tf.sparse_placeholder(tf.int32,name='targets')

        lstm_initializer = tf.contrib.layers.xavier_initializer()
        fw_layer = layers.lstm(BRNN['layers'],BRNN['hidden_units'],lstm_initializer,dropout=dropout_lstm)
        bw_layer = layers.lstm(BRNN['layers'],BRNN['hidden_units'],lstm_initializer,dropout=dropout_lstm)
        (outputs_fw,outputs_bw),_ = tf.nn.bidirectional_dynamic_rnn(fw_layer,bw_layer,lstm_inputs,dtype=tf.float32)

        # outputs,_,_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,cells_bw,inputs = lstm_inputs,dtype=tf.float32)

        if debug:
            print('LSTM_Output_size:({},{})'.format(outputs_fw,outputs_bw))

        #Concatenate the output from both cells (forward and backward)
        blstm_outputs = tf.concat([outputs_fw,outputs_bw], 2)

        #flatten out all except the last dimension
        fc_inputs  = tf.reshape(blstm_outputs,[-1,2*BRNN['hidden_units']])

        #Feed into the fully connected layer
        #No activation cuz, the output of this layer is feeded into CTC Layer as logits
        for i in range(len(wfc)):
            fc_out = layers.fc(fc_inputs,wfc[i],bfc[i],activation=None,dropout=dropout_fc)
            #Input to next layer is output of previous layer..
            fc_inputs = fc_out

        #Reshape back to batch_size, seq_len,vocab_size
        logits = tf.reshape(fc_out,[-1,seq_len,vocab_size])

        #convert them to time major
        logits = tf.transpose(logits,[1,0,2])

        #Calculate loss
        loss = tf.reduce_mean(tf.nn.ctc_loss(targets, logits, time_steps))

        #Optimize
        optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha)
        train = optimizer.minimize(loss)

        # CTC decoder.
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, time_steps)
        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))

        return graph,dropout_conv,dropout_lstm,dropout_fc,inputs,time_steps,targets,loss,train,decoded,label_error_rate,seq_len