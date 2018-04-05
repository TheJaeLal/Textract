import tensorflow as tf
from math import ceil

def fc(x,w,b,activation=None,dropout=1.0):
    
    out = tf.matmul(x,w) + b
    
    if activation == 'relu':
        out = tf.nn.relu(out)
    
    elif activation == 'leaky_relu':
        out = tf.nn.leaky_relu(out)
    
    elif activation == 'elu':
        out = tf.nn.elu(out)
        
    elif activation == 'tanh':
        out = tf.nn.elu(out)
        
    out = tf.nn.dropout(out,dropout)
    
    return out

def lstm(num_layers,hidden_units,initializer,dropout=1.0):
    
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.LSTMCell(hidden_units,initializer=initializer)
        cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=dropout,dtype=tf.float32)
        cells.append(cell)
    
    layer = tf.contrib.rnn.MultiRNNCell(cells)
    
    return layer

def conv(x,w,b,s,activation=None,dropout=1.0):
    conv = tf.nn.conv2d(input=x,filter=w,padding='SAME',strides=[1,s,s,1]) + b
    
    if activation == 'relu':
        conv = tf.nn.relu(conv)
    
    elif activation == 'leaky_relu':
        conv = tf.nn.leaky_relu(conv)
        
    elif activation == 'elu':
        conv = tf.nn.elu(conv)
        
    elif activation == 'tanh':
        conv = tf.nn.tanh(conv)
        
    conv = tf.nn.dropout(conv,dropout)
    
    return conv

def max_pool(x,s):
    
    pool = tf.nn.max_pool(x,ksize=[1,s,s,1],strides=[1,s,s,1],padding='SAME')
    
    return pool

def calc_out_dims(CNN,height,width):
    for i in range(len(CNN)):

        strides = CNN[i]['conv'][1]

        height = ceil(float(height) / float(strides))
        width = ceil(float(width) / float(strides))

        if CNN[i]['pool']:
            height = ceil(float(height) / float(CNN[i]['pool']))
            width = ceil(float(width) / float(CNN[i]['pool']))    
    
    return height,width

def init_weights(CNN,FC): 
    
        
    #Weights Initializer
    fc_initializer = tf.contrib.layers.xavier_initializer()
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()

    wconv_shapes = [] ; bconv_shape = []
    wfc_shapes = []; bfc_shapes = []

    #All theses params are created and returned...
    wconv = [] ; bconv = []
    wfc = [] ; bfc = []

    #Setup the conv_filter shapes
    for i in range(len(CNN)):
        # i goes from 0 to 4
        # i+1 goes from 1 to 5

        filter_size = CNN[i]['conv'][0]

        if i == 0:
            #Number of input channels = 1 in 1st conv layer
            ch_in = 1
        else:
            #Output channels/filters of the previous layer..
            ch_in = CNN[i-1]['conv'][2]

        #Number of output_channels/filters
        ch_out = CNN[i]['conv'][2]

        wconv_shapes.append([filter_size,filter_size,ch_in,ch_out])

    #All right here...
    #print('wconv_shapes') 
    #print(wconv_shapes)

    #Setup Fully connected weight shapes..
    for i in range(1,len(FC)):
        wfc_shapes.append([ FC[i-1]['units'], FC[i]['units'] ])
        bfc_shapes.append([ FC[i]['units'] ])

    #All right here...    
    #print('Fully connected shapes')
    #print(wfc_shapes)
    #print(bfc_shapes)

    #Create Weights and Biases
    for i in range(len(CNN)):
        wconv.append(tf.Variable(conv_initializer(wconv_shapes[i])))
        bconv.append(tf.Variable(tf.zeros(bconv_shape))) 

    #All right here..
    #print(wconv)
    #print(bconv)
    #print(len(wfc_shapes))
    #print(len(bfc_shapes))

    for i in range(len(FC)-1):
        wfc.append(tf.Variable(fc_initializer(wfc_shapes[i])))
        bfc.append(tf.Variable(tf.zeros(bfc_shapes[i])))

    return wconv,bconv,wfc,bfc