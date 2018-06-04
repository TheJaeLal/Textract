import tensorflow as tf
#from tensorflow.python.client import device_lib
from math import ceil

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

# def train_mode(tensor,gamma,beta,avg_mean,avg_variance,momentum,epsilon):

#     tensor,mean,variance = tf.nn.fused_batch_norm(tensor, scale=gamma, offset=beta,
#                                                   epsilon=epsilon, is_training=True)
    
#     update_mean = tf.assign(avg_mean, avg_mean * momentum + mean * (1.0 - momentum))
#     update_variance = tf.assign(avg_variance, avg_variance * momentum + variance * (1.0 - momentum))
#     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
#     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)

#     return tensor,mean,variance

# def infer_mode(tensor,gamma,beta,avg_mean,avg_variance,epsilon):
    
#     return tf.nn.fused_batch_norm(tensor, scale=gamma, offset=beta, mean=avg_mean,
#                                         variance=avg_variance, epsilon=epsilon, is_training=False) 

        
# def batch_norm(tensor,scale,is_training,epsilon=0.001, momentum=0.9, fused_batch_norm=True, name=None):
    
#     """Performs batch normalization on given 4-D tensor.
#     The features are assumed to be in NHWC format. Note that you need to
#     run UPDATE_OPS in order for this function to perform correctly, e.g.:
#     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#       train_op = optimizer.minimize(loss)
#     Based on: https://arxiv.org/abs/1502.03167
#     """
    
#     with tf.variable_scope(name, default_name='batch_norm'):

#         shapes = tensor.get_shape()
        
#         channels = shapes[-1]

#         axes = list(range(len(shapes)-1))

#         beta = tf.get_variable('beta', channels, initializer=tf.zeros_initializer())

#         if scale:
#             gamma = tf.get_variable('gamma', channels, initializer=tf.ones_initializer())
#         else:
#             gamma = tf.get_variable('gamma', channels, initializer=tf.ones_initializer(),trainable=False)

#         avg_mean = tf.get_variable('avg_mean', channels, initializer=tf.zeros_initializer(), trainable=False)
#         avg_variance = tf.get_variable('avg_variance', channels, initializer=tf.ones_initializer(), trainable=False)

#         tensor,mean,variance = tf.cond(is_training,lambda: train_mode(tensor,gamma,beta,
#                                                                      avg_mean,avg_variance,momentum,epsilon),
#                                               lambda: infer_mode(tensor,gamma,beta,avg_mean,avg_variance,epsilon))
        
        
#         return tensor

def fc(x,units,activation=None,is_training=False,dropout=1.0):
    
    #print('**Fully connected info:')
    #print('units = ',units)
    #print('activation = ',activation)
#     print('use_bn = ',use_bn)
    #print('is_training = ',is_training)
    #print('dropout = ',dropout)
    
    x = tf.nn.dropout(x,keep_prob = dropout)
    #out = tf.matmul(x,w)
    fc_initializer = tf.contrib.layers.xavier_initializer()
        
    out = tf.layers.dense(x,units,activation=None,use_bias=True,kernel_initializer=fc_initializer)
    
#     tf.summary.histogram('fc_logits',out)

#     print(out.get_shape())
    #Scale is set to True cuz, mostly we're never gonna use activations here..
#     out = tf.contrib.layers.batch_norm(out,axis=1,scale=True,is_training=is_training,fused=None)
#     out = tf.layers.batch_normalization(out,axis=-1,training=is_training,fused=None)    
        
    if activation == 'relu':
        out = tf.nn.relu(out)
    
    elif activation == 'leaky_relu':
        out = tf.nn.leaky_relu(out)
    
    elif activation == 'elu':
        out = tf.nn.elu(out)
        
    elif activation == 'tanh':
        out = tf.nn.tanh(out)
    
    tf.summary.histogram('fc_activations',out)
    
    #The value passed is the keep_prob!!
#    dropout = 1-dropout
    
    #Apply Dropout...
#    out = tf.layers.dropout(out,dropout,training=is_training)
    
#     tf.summary.histogram('fc_dropout',out)
    
#     out = tf.nn.dropout(out,dropout)
    
    return out

def lstm(layers,clip,initializer,dropout=1.0):
    
#--> Try progressively reducing the number of layers
    cells = []

#    gpus = get_available_gpus()
#    num_gpus = len(gpus)
    
    for i in range(len(layers)):
#         for i in range(num_layers):
#         cell = tf.contrib.rnn.LSTMCell(layers[i])
        
        cell = tf.contrib.rnn.LSTMBlockCell(layers[i],cell_clip=clip)
#         cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=dropout,dtype=tf.float32)        

        #Palce on multiple gpus
#        cell = tf.contrib.rnn.DeviceWrapper(cell,"/gpu:%d" % (i % num_gpus))
#         cell = tf.contrib.rnn.ResidualWrapper(cell)
        cells.append(cell)
    
#     layer = tf.contrib.rnn.MultiRNNCell(cells)    
        
    return cells

def conv(x,nf,k,s,padding,activation=None,is_training=False,dropout=1.0,use_batchnorm=False):
    
    #print('++Conv layer info info:')
#     print('use_bn = ',use_bn)
    #print('Num_filters = ',nf)
    #print('Kernel_size = ',k)
    #print('Strides = ',s)
    #print('activation = ',activation)
    #print('is_training = ',is_training)
    #print('dropout = ',dropout)
        
#     conv = tf.nn.conv2d(input=x,filter=w,padding='SAME',strides=[1,s,s,1])      
#     conv = tf.contrib.layers.batch_norm(conv,scale=scale,is_training=is_training,fused=None)

    x = tf.nn.dropout(x,keep_prob = dropout)

    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    if activation == 'relu':
        scale = False
    else:
        scale = True

    #No bias because we want to removing padding after convolution..
    #Doesn't matter, I guess, use bias anyway...
    
    use_bias_bool = True
    
    if use_batchnorm:
        use_bias_bool = False
    
    conv = tf.layers.conv2d(x,nf,k,s,padding=padding,activation=None,use_bias=use_bias_bool,
                            kernel_initializer=conv_initializer)

#     tf.summary.histogram('conv_logits',conv)

    if use_batchnorm:
        conv = tf.layers.batch_normalization(conv,scale=scale,training=is_training,fused=None)
#     conv = batch_norm(conv,scale,is_training)
#     tf.summary.histogram('conv_bn',conv)
    
    if activation == 'relu':
        conv = tf.nn.relu(conv)
    
    elif activation == 'leaky_relu':
        conv = tf.nn.leaky_relu(conv)
        
    elif activation == 'elu':
        conv = tf.nn.elu(conv)
        
    elif activation == 'tanh':
        conv = tf.nn.tanh(conv)
    
    tf.summary.histogram('conv_activations',conv)
    
#     conv = tf.nn.l2_normalize(conv,[1,2])

    #The value passed is the keep_prob
#    dropout = 1-dropout             
#    conv = tf.layers.dropout(conv,dropout,training = is_training)
    
#     conv = tf.nn.dropout(conv,dropout)
    
    return conv

def max_pool(x,s):
    
    #print('--Max_pool info:')
    #print('Strides = ',s)
    pool = tf.layers.max_pooling2d(x,s,s,padding='SAME')
#     pool = tf.nn.max_pool(x,ksize=[1,s,s,1],strides=[1,s,s,1],padding='SAME')

    return pool

def calc_out_dims(CNN,height,width):
    for i in range(len(CNN)):

        strides = CNN[i]['conv'][1]
        kernel_size = CNN[i]['conv'][0]
        
        #Same Padding
        if CNN[i]['padding'] == 'SAME':
            height = ceil(float(height) / float(strides))
            width = ceil(float(width) / float(strides))
        
        #Valid Padding
        else:
            height = ceil(float(height) - float(kernel_size) / float(strides)) + 1
            width = ceil(float(width)  - float(kernel_size) / float(strides)) + 1
            
            
        if CNN[i]['pool']:
            height = ceil(float(height) / float(CNN[i]['pool'][0]))
            width = ceil(float(width) / float(CNN[i]['pool'][1]))    
    
    return height,width