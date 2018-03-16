# Copyright 2017 Zhongyi Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 512 SpinePathNet network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
#from tensorflow.contrib import rnn
from collections import namedtuple
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
    
def weight_variable(shape):
    #initializer = tf.truncated_normal_initializer()
    initializer= tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float16)
    return tf.get_variable("weights", shape, initializer=initializer)
    
def bias_variable(shape):
    initializer=tf.constant_initializer(0.1)
    return tf.get_variable("biases", shape, initializer=initializer)
    
def conv_layer(x, w, b, name, padding = 'SAME'):
    w = weight_variable(w)
    b = bias_variable(b)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding = padding, name = name) + b

def conv_layer_gan(x, w, b, name, padding = 'SAME'):
    w = weight_variable(w)
    b = bias_variable(b)
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding = padding, name = name) + b

def deconv_layer(x,w,output_shape,name=None):
    
    w = weight_variable(w)    
    return tf.nn.conv2d_transpose(x,w,output_shape,strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC',name=name)
    
def batch_normalization(inputs, scope, is_training=True):
    bn = tf.contrib.layers.batch_norm(inputs,
        decay=0.999,
        center=True,
        scale=False,
        epsilon=0.001,
        activation_fn=None,
        param_initializers=None,
        param_regularizers=None,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=is_training,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        batch_weights=None,
        fused=False,
        data_format='NHWC',
        zero_debias_moving_mean=False,
        scope=scope,
        renorm=False,
        renorm_clipping=None,
        renorm_decay=0.99)
    return tf.nn.relu(bn, name='relu')
        
def maxpooling_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name = name)

def avgpooling_2x2(x,name):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name = name)

def maxpooling_4x4(x,name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='SAME',name = name)

def avgpooling_4x4(x):
    return tf.nn.avg_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    
def atrous_conv_layer(x, w, b, rate, name, padding = 'SAME'):
    w = weight_variable(w)
    b = bias_variable(b)
    return tf.nn.atrous_conv2d(x, w, rate, padding, name = name) + b  #without stride


def biLSTM(logits, class_num, batch_size):
    
    n_classes = class_num
    n_steps = 512*512
    n_hidden = n_classes
    
    
    logits = tf.reshape(logits, [batch_size,-1, n_classes])
    
    x = tf.unstack(logits, n_steps, 1)
    
    # Forward direction cell
    lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0)    
    # Backward direction cell
    #lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    #outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
    #                                          dtype=tf.float32)
    outputs, _,  = rnn.static_rnn(lstm_fw_cell, x,
                                              dtype=tf.float32)
    return outputs

def gridLSTM1x1(feature, feature_size, feature_dimension, batch_size):
    
   
    n_steps = feature_size*feature_size
    n_hidden = feature_dimension
    
    
    feature = tf.reshape(feature, [batch_size, -1, feature_size])
    
    x = tf.unstack(feature, n_steps, 1)
    
    # Forward direction cell
    #lstm_fw_cell = tf.contrib.rnn.GridLSTMCell(n_hidden, forget_bias=1.0)    
    # Backward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    #outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
    #                                          dtype=tf.float32)
    outputs, _,  = rnn.static_rnn(lstm_fw_cell, x, dtype=tf.float32)
    
    outputs = tf.reshape(outputs,[batch_size, feature_size,feature_size,feature_dimension])
    print (outputs.shape)
    return outputs


def LSTM2x2(net, superpixel, feature_size, feature_dimension, batch_size):
    
    #n_steps is the number of superpixels.
    n_steps = 2000
    n_hidden = feature_dimension
    indexs_batch = []
    one_node_features = []
    #pooling
    for j in xrange(batch_size):
        indexs_image = []
        for i in xrange(n_steps):        
            indexs = tf.where(tf.equal(superpixel[j,:,:],i))        
            indexs_image.append(indexs)        
            one_node_feature = tf.gather_nd(net[j,:,:,:],indexs)            
            one_node_feature = tf.reduce_mean(one_node_feature,axis=0, keep_dims=False)                 
            one_node_features.append(one_node_feature)    
        indexs_batch.append(indexs_image)
    one_node_features = tf.reshape(one_node_features, [batch_size, -1, 128])
    #print (one_node_features.shape) #shape is  [batch size, 2000, 128].  
    features = tf.unstack(one_node_features, n_steps, 1)
    
    
    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
    
    #lstm is a 2000 length list, the shape of per element is (4,128).ok.
    lstm, _ = tf.contrib.rnn.static_rnn(lstm_cell, features, dtype=tf.float32) 
    
    """ 
    unpooling, restore results of net.
    indexs length is 8000. the shape of per element is (?,2). 
    ? is the unkown nomber of pixels belonging to one superpixel.
    2 is the coordate (x,y) of one batch.
    """
    nets = []
    for j in xrange(4):       
        net_one_image = net[j,:,:,:]
        for i in xrange(n_steps):  #n_steps          
            indices = indexs_batch[j][i]# The j-th batch and i-th superpixel.                    
            updates = tf.expand_dims(lstm[i][j,:],0) 
            #update = updates
            updates = tf.tile(updates,[tf.div(tf.size(indices),2),1])
            print (updates.shape)
            #x = 0
            #while tf.less_equal(x, tf.div(tf.size(indices),2) ) is True:
            #for z in xrange(300):
             #   update = tf.concat([update,updates],0)
            #    x = x + 1
            #scatter = 
            net_one_image = net_one_image  + tf.scatter_nd(indices, updates, tf.constant([512,512,128], dtype=tf.int64))  
        nets.append(net_one_image)

    return net + tf.to_float(tf.reshape(nets, [batch_size, 512, 512, 128]))


def LSTM_pool(net, feature_size, feature_dimension, batch_size, is_training):
    
    net = avgpooling_4x4(net)
    n_hidden = feature_dimension
    net = tf.reshape(net, [batch_size, -1, feature_dimension])
    n_steps = net.shape[1]    
    net = tf.unstack(net, n_steps, 1)
    fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True, forget_bias=1.0)
    #bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True, forget_bias=1.0)
    net, _, = tf.contrib.rnn.static_rnn(fw_cell, net, dtype=tf.float32) 
    net = tf.reshape(net, [batch_size, 32, 32, feature_dimension])    
    with tf.variable_scope('deconv_3'):
        net = deconv_layer(net,[3,3,128,128],[batch_size,64,64,128], name = 'g_deconv_3')
        net = batch_normalization(net,'g_bn_11', is_training=is_training)
    with tf.variable_scope('deconv_4'):
        net = deconv_layer(net,[3,3,128,128],[batch_size,128,128,128], name = 'g_deconv_4')
        net = batch_normalization(net,'g_bn_12', is_training=is_training)
    return net



def rnn(net, feature_size, feature_dimension, batch_size):
    
    patch_size = 4
   
    #n_steps = feature_size*feature_size
    n_hidden = feature_dimension*patch_size
    
    
    net = tf.reshape(net, [batch_size, -1, feature_dimension*patch_size])
    
    print (net.shape)
    
    n_steps = net.shape[1]
    
    net = tf.unstack(net, n_steps, 1)
    #print (tf.size(x))
    # Forward direction cell
    #lstm_fw_cell = tf.contrib.rnn.GridLSTMCell(n_hidden, forget_bias=1.0)    
    # Backward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

    net, _,  = tf.contrib.rnn.static_rnn(lstm_fw_cell, net, dtype=tf.float32)

    #print (outputs.shape)
    return tf.reshape(net,[batch_size, feature_size,feature_size,feature_dimension])
                   
# =========================================================================== #
# SpinePathNet build.
# =========================================================================== #
#Noly use for basic convolutional network.
def net(inputs, batch_size, class_num, reuse=False,
             is_training=True, scope='SpinePathNet'):
    
    with tf.variable_scope(scope, 'SpinePathNet',[input], reuse=reuse):
        with tf.variable_scope('conv_1'):
            conv_1 = conv_layer(inputs, [7,7,1,32], 32, 'conv_1')#receptive field = 7
            bn_1 = batch_normalization(conv_1,'bn_1', is_training=is_training)                
                
        with tf.variable_scope('conv_2'):
            conv_2 = conv_layer(bn_1, [7,7,32,32], 32, 'conv_2')#receptive field = 13
            bn_2 = batch_normalization(conv_2,'bn_2', is_training=is_training)
            pool_conv2 = maxpooling_2x2(bn_2, 'pool_conv2')   #receptive field = 26  
                
        with tf.variable_scope('conv_3'):
            conv_3 = conv_layer(pool_conv2, [3,3,32,64], 64, 'conv_3')#receptive field = 28]
            bn_3 = batch_normalization(conv_3,'bn_3', is_training=is_training)                         
            
        with tf.variable_scope('conv_4'):
            conv_4 = conv_layer(bn_3, [3,3,64,64], 64, 'conv_4') #receptive field = 30
            bn_4 = batch_normalization(conv_4,'bn_4', is_training=is_training)                
            pool_conv4= maxpooling_2x2(bn_4, 'pool_conv3')   #receptive field = 60
            
        with tf.variable_scope('conv_5'):
            conv_5 = atrous_conv_layer(pool_conv4, [3,3,64,128], 128, 2, 'conv_5')#receptive field = 66
            bn_5 = batch_normalization(conv_5,'bn_5', is_training=is_training)
            
        with tf.variable_scope('conv_6'):
            conv_6 = atrous_conv_layer(bn_5, [3,3,128,128], 128, 4, 'conv_6')#receptive field = 76
            bn_6 = batch_normalization(conv_6,'bn_6', is_training=is_training)            
            
        with tf.variable_scope('conv_7'):
            conv_7 = atrous_conv_layer(bn_6, [3,3,128,128], 128, 8, 'conv_7')#receptive field = 94
            bn_7 = batch_normalization(conv_7, 'bn_7', is_training=is_training)
            
        with tf.variable_scope('conv_8'):
            conv_8 = atrous_conv_layer(bn_7, [3,3,128,128], 128, 16, 'conv_8')#receptive field = 128
            bn_8 = batch_normalization(conv_8, 'bn_8', is_training=is_training)
                
        with tf.variable_scope('deconv_1'): 
            deconv_1 = deconv_layer(bn_8,[3,3,128,128],[batch_size,256,256,128], name = 'deconv_1')
            bn_9 = batch_normalization(deconv_1, 'bn_9', is_training=is_training)
            
        with tf.variable_scope('deconv_2'): 
            deconv_2 = deconv_layer(bn_9,[3,3,128,128],[batch_size,512,512,128],name='deconv_2')
            bn_10 = batch_normalization(deconv_2, 'bn_10', is_training=is_training)
                        
        with tf.variable_scope('feature_embedding'):            
            conv_9 = conv_layer(bn_10, [1,1,128,64], 64, 'feature_embedding')
            feature_embedding = batch_normalization(conv_9, 'feature_embedding', is_training=is_training)
            
        with tf.variable_scope('logits'):
            logits = conv_layer(feature_embedding, [1,1,64,class_num],class_num,'logits')        
            
    return feature_embedding, logits  


#Noly use for basic convolutional network and adverisal network. without lstm.
def g_net(inputs, batch_size, class_num, reuse=False,
             is_training=True, scope='g_SpinePathNet'):
    
    with tf.variable_scope(scope, 'g_SpinePathNet',[input], reuse=reuse):
        with tf.variable_scope('conv_1'):
            conv_1 = conv_layer(inputs, [7,7,1,32], 32, 'g_conv_1')#receptive field = 7
            bn_1 = batch_normalization(conv_1,'g_bn_1', is_training=is_training)                
                
        with tf.variable_scope('conv_2'):
            conv_2 = conv_layer(bn_1, [7,7,32,32], 32, 'g_conv_2')#receptive field = 13
            bn_2 = batch_normalization(conv_2,'g_bn_2', is_training=is_training)
            pool_conv2 = maxpooling_2x2(bn_2, 'g_pool_conv2')   #receptive field = 26  
                
        with tf.variable_scope('conv_3'):
            conv_3 = conv_layer(pool_conv2, [3,3,32,64], 64, 'g_conv_3')#receptive field = 28]
            bn_3 = batch_normalization(conv_3,'g_bn_3', is_training=is_training)                         
            
        with tf.variable_scope('conv_4'):
            conv_4 = conv_layer(bn_3, [3,3,64,64], 64, 'g_conv_4') #receptive field = 30
            bn_4 = batch_normalization(conv_4,'g_bn_4', is_training=is_training)                
            pool_conv4= maxpooling_2x2(bn_4, 'g_pool_conv3')   #receptive field = 60
            
        with tf.variable_scope('conv_5'):
            conv_5 = atrous_conv_layer(pool_conv4, [3,3,64,128], 128, 2, 'g_conv_5')#receptive field = 66
            bn_5 = batch_normalization(conv_5,'g_bn_5', is_training=is_training)
            
        with tf.variable_scope('conv_6'):
            conv_6 = atrous_conv_layer(bn_5, [3,3,128,128], 128, 4, 'g_conv_6')#receptive field = 76
            bn_6 = batch_normalization(conv_6,'g_bn_6', is_training=is_training)            
            
        with tf.variable_scope('conv_7'):
            conv_7 = atrous_conv_layer(bn_6, [3,3,128,128], 128, 8, 'g_conv_7')#receptive field = 94
            bn_7 = batch_normalization(conv_7, 'g_bn_7', is_training=is_training)
            
        with tf.variable_scope('conv_8'):
            conv_8 = atrous_conv_layer(bn_7, [3,3,128,128], 128, 16, 'g_conv_8')#receptive field = 128
            bn_8 = batch_normalization(conv_8, 'g_bn_8', is_training=is_training)
                
        with tf.variable_scope('deconv_1'): 
            deconv_1 = deconv_layer(bn_8,[3,3,128,128],[batch_size,256,256,128], name = 'g_deconv_1')
            bn_9 = batch_normalization(deconv_1, 'g_bn_9', is_training=is_training)
            
        with tf.variable_scope('deconv_2'): 
            deconv_2 = deconv_layer(bn_9,[3,3,128,128],[batch_size,512,512,128],name='g_deconv_2')
            bn_10 = batch_normalization(deconv_2, 'g_bn_10', is_training=is_training)
                        
        with tf.variable_scope('feature_embedding'):            
            conv_9 = conv_layer(bn_10, [1,1,128,64], 64, 'g_feature_embedding')
            feature_embedding = batch_normalization(conv_9, 'g_feature', is_training=is_training)
            
        with tf.variable_scope('logits'):
            logits = conv_layer(feature_embedding, [1,1,64,class_num],class_num,'g_logits')
                    
    return deconv_1, logits    




#Use for basic convolutional network + adverisal network + lstm.
def g_l_net(inputs, batch_size, class_num, reuse=False, is_training=True, scope='g_SpinePathNet'):
    
    with tf.variable_scope(scope, 'g_SpinePathNet',[input], reuse=reuse):
        with tf.variable_scope('conv_1'):
            net = conv_layer(inputs, [7,7,1,32], 32, 'g_conv_1')#receptive field = 7
            net = batch_normalization(net,'g_bn_1', is_training=is_training)                
                
        with tf.variable_scope('conv_2'):
            net = conv_layer(net, [7,7,32,32], 32, 'g_conv_2')#receptive field = 13
            net = batch_normalization(net,'g_bn_2', is_training=is_training)
            net = maxpooling_2x2(net, 'g_pool_conv2')   #receptive field = 26  
                
        with tf.variable_scope('conv_3'):
            net = conv_layer(net, [3,3,32,64], 64, 'g_conv_3')#receptive field = 28]
            net = batch_normalization(net,'g_bn_3', is_training=is_training)                         
            
        with tf.variable_scope('conv_4'):
            net = conv_layer(net, [3,3,64,64], 64, 'g_conv_4') #receptive field = 30
            net = batch_normalization(net,'g_bn_4', is_training=is_training)                
            net = maxpooling_2x2(net, 'g_pool_conv3')   #receptive field = 60
            
        with tf.variable_scope('conv_5'):
            net = atrous_conv_layer(net, [3,3,64,128], 128, 2, 'g_conv_5')#receptive field = 66
            net = batch_normalization(net,'g_bn_5', is_training=is_training)
            
        with tf.variable_scope('conv_6'):
            net = atrous_conv_layer(net, [3,3,128,128], 128, 4, 'g_conv_6')#receptive field = 76
            net = batch_normalization(net,'g_bn_6', is_training=is_training)            
            
        with tf.variable_scope('conv_7'):
            net = atrous_conv_layer(net, [3,3,128,128], 128, 8, 'g_conv_7')#receptive field = 94
            net = batch_normalization(net, 'g_bn_7', is_training=is_training)
            
        with tf.variable_scope('conv_8'):
            net = atrous_conv_layer(net, [3,3,128,128], 128, 16, 'g_conv_8')#receptive field = 128
            net_1 = batch_normalization(net, 'g_bn_8', is_training=is_training)
            
        with tf.variable_scope('LSTM'):
            net_2 = LSTM_pool(net_1, 128, 128, batch_size, is_training=is_training)
                
        with tf.variable_scope('deconv_1'): 
            net_3 = deconv_layer(net_1+net_2,[3,3,128,128],[batch_size,256,256,128], name = 'g_deconv_1')
            net = batch_normalization(net_3, 'g_bn_9', is_training=is_training)
            
        with tf.variable_scope('deconv_2'): 
            net = deconv_layer(net,[3,3,128,128],[batch_size,512,512,128],name='g_deconv_2')
            net = batch_normalization(net, 'g_bn_10', is_training=is_training)
            
       # with tf.variable_scope('LSTM'):
        #    net = LSTM2x2(net, superpixel, 512, 128, batch_size) 
            #net2 = batch_normalization(net, 'g_bn_10', is_training=is_training)
                        
        with tf.variable_scope('conv_9'):            
            net = conv_layer(net, [1,1,128, 64], 64, 'g_conv_9')
            net = batch_normalization(net, 'g_feature', is_training=is_training)
            
        with tf.variable_scope('logits'):
            net = conv_layer(net, [1,1,64,class_num], class_num,'g_logits')                  
            
    return net,net_1,net_3     





def d_net(inputs, class_num, reuse = False, is_training=True, scope='d_gan'):
    
    with tf.variable_scope(scope, 'd_gan', [input], reuse=reuse):
        
        with tf.variable_scope('gan_conv_1'):
            net = conv_layer_gan(inputs, [7,7,class_num,32], 32, 'd_conv_1')
            net = batch_normalization(net,'d_bn_1', is_training=is_training)   
            net = avgpooling_2x2(net, 'pool_conv1')
            
        with tf.variable_scope('gan_conv_2'):
            net = conv_layer_gan(net, [7,7,32,64], 64, 'd_conv_2')
            net = batch_normalization(net,'d_bn_2', is_training=is_training)   
            net = avgpooling_2x2(net, 'pool_conv2')
            
        with tf.variable_scope('gan_conv_3'):
            net = conv_layer_gan(net, [7,7,64,128], 128, 'd_conv_3')
            net = batch_normalization(net,'d_bn_3', is_training=is_training)   
            net = avgpooling_2x2(net, 'pool_conv3')
            
        with tf.variable_scope('d_fc_1'):
            net = tf.contrib.layers.fully_connected(net,256)
            net =  tf.contrib.layers.dropout(net, keep_prob=0.6)
            
        with tf.variable_scope('d_output'):
            
            net = tf.contrib.layers.fully_connected(net,1,activation_fn=None)
            
    return net
            
          

# define u-net for comparison.
def unet(inputs, batch_size, class_num, reuse=False,
             is_training=True, scope='unet'):
    
    with tf.variable_scope(scope, 'unet',[input], reuse=reuse):
        with tf.variable_scope('conv_1'):
            net = conv_layer(inputs, [3,3,1,64], 64, 'g_conv_1')#receptive field = 7
            net = tf.nn.relu(net)                
                
        with tf.variable_scope('conv_2'):
            net = conv_layer(net, [3,3,64,64], 64, 'g_conv_2')#receptive field = 13
            net_1 = tf.nn.relu(net) 
            net = maxpooling_2x2(net_1, 'g_pool_conv2')   #receptive field = 26  
                
        with tf.variable_scope('conv_3'):
            net = conv_layer(net, [3,3,64,128], 128, 'g_conv_3')#receptive field = 28]
            net = tf.nn.relu(net)                     
            
        with tf.variable_scope('conv_4'):
            net = conv_layer(net, [3,3,128,128], 128, 'g_conv_4') #receptive field = 30
            net_2 = tf.nn.relu(net)               
            net = maxpooling_2x2(net_2, 'g_pool_conv4')   #receptive field = 60
            
        with tf.variable_scope('conv_5'):
            net = conv_layer(net, [3,3,128,256], 256, 'g_conv_5')#receptive field = 28]
            net = tf.nn.relu(net)  
            
        with tf.variable_scope('conv_6'):
            net = conv_layer(net, [3,3,256,256], 256, 'g_conv_6') #receptive field = 30
            net_3 = tf.nn.relu(net)               
            net = maxpooling_2x2(net_3, 'g_pool_conv6')   #receptive field = 60         
            
        with tf.variable_scope('conv_7'):
            net = conv_layer(net, [3,3,256,512], 512, 'g_conv_7')#receptive field = 28]
            net = tf.nn.relu(net)
            
        with tf.variable_scope('conv_8'):
            net = conv_layer(net, [3,3,512,512], 512, 'g_conv_8') #receptive field = 30
            net_4 = tf.nn.relu(net)               
            net = maxpooling_2x2(net_4, 'g_pool_conv8')   #receptive field = 60
            
        with tf.variable_scope('conv_9'):
            net = conv_layer(net, [3,3,512,1024], 1024, 'g_conv_9')#receptive field = 28]
            net = tf.nn.relu(net)
        
        with tf.variable_scope('conv_10'):
            net = conv_layer(net, [3,3,1024,512], 512, 'g_conv_10')#receptive field = 28]
            net = tf.nn.relu(net)
                
        with tf.variable_scope('deconv_1'): 
            net = deconv_layer(net,[2,2,512,512],[batch_size,64,64,512], name = 'g_deconv_1')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_1_conv_1'): 
            net = conv_layer(tf.concat([net_4,net],-1), [3,3,1024,512], 512, 'deconv_1_conv_1')#receptive field = 28]
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_1_conv_2'): 
            net = conv_layer(net, [3,3,512,256], 256, 'deconv_1_conv_2')#receptive field = 28]
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_2'): 
            net = deconv_layer(net,[2,2,256,256],[batch_size,128,128,256], name = 'g_deconv_2')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_2_conv_1'): 
            net = conv_layer(tf.concat([net_3,net],-1), [3,3,512,256], 256, 'deconv_2_conv_1')#receptive field = 28]
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_2_conv_2'): 
            net = conv_layer(net, [3,3,256,128], 128, 'deconv_2_conv_2')#receptive field = 28]
            net = tf.nn.relu(net)
        
        with tf.variable_scope('deconv_3'): 
            net = deconv_layer(net,[2,2,128,128],[batch_size,256,256,128], name = 'g_deconv_1')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_3_conv_1'): 
            net = conv_layer(tf.concat([net_2,net],-1), [3,3,256,128], 128, 'deconv_3_conv_1')#receptive field = 28]
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_3_conv_2'): 
            net = conv_layer(net, [3,3,128,64], 64, 'deconv_3_conv_2')#receptive field = 28]
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_4'): 
            net = deconv_layer(net,[2,2,64,64],[batch_size,512,512,64], name = 'g_deconv_4')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_4_conv_1'): 
            net = conv_layer(tf.concat([net_1,net],-1), [3,3,128,64], 64, 'deconv_4_conv_1')#receptive field = 28]
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_4_conv_2'): 
            net = conv_layer(net, [3,3,64,64], 64, 'deconv_4_conv_2')#receptive field = 28]
            net = tf.nn.relu(net)

        with tf.variable_scope('logits'):
            net = conv_layer(net, [1,1,64,class_num], class_num,'g_logits')                  
            
    return net    

# define u-net for comparison.
def vgg16_1(inputs, batch_size, class_num, reuse=False,
             is_training=True, scope='vgg_16'):
    
    with tf.variable_scope(scope, 'vgg_16',[input], reuse=reuse):
        with tf.variable_scope('conv1/conv1_1'):
            net = conv_layer(inputs, [3,3,1,64], 64, 'conv1/conv1_1')
            net = tf.nn.relu(net)
        with tf.variable_scope('conv1/conv1_2'):
            net = conv_layer(net, [3,3,64,64], 64, 'conv1/conv1_2')
            net = tf.nn.relu(net)                
            net = maxpooling_2x2(net, 'pool1')     
            
        with tf.variable_scope('conv2/conv2_1'):
            net = conv_layer(net, [3,3,64,128], 128, 'conv2/conv2_1')
            net = tf.nn.relu(net)
        with tf.variable_scope('conv2/conv2_2'):
            net = conv_layer(net, [3,3,128,128], 128, 'conv2/conv2_2')
            net = tf.nn.relu(net)                
            net = maxpooling_2x2(net, 'pool2')
                
        with tf.variable_scope('conv3/conv3_1'):
            net = conv_layer(net, [3,3,128,256], 256, 'conv3/conv3_1')
            net = tf.nn.relu(net)
        with tf.variable_scope('conv3/conv3_2'):
            net = conv_layer(net, [3,3,256,256], 256, 'conv3/conv3_2')
            net = tf.nn.relu(net)  
        with tf.variable_scope('conv3/conv3_3'):
            net = conv_layer(net, [3,3,256,256], 256, 'conv3/conv3_3')
            net = tf.nn.relu(net)  
            net = maxpooling_2x2(net, 'pool3')
            
        with tf.variable_scope('conv4/conv4_1'):
            net = conv_layer(net, [3,3,256,512], 512, 'conv4/conv4_1')
            net = tf.nn.relu(net)
        with tf.variable_scope('conv4/conv4_2'):
            net = conv_layer(net, [3,3,512,512], 512, 'conv4/conv4_2')
            net = tf.nn.relu(net) 
        with tf.variable_scope('conv4/conv4_3'):
            net = conv_layer(net, [3,3,512,512], 512, 'conv4/conv4_3')
            net = tf.nn.relu(net)  
            net = maxpooling_2x2(net, 'pool4')
            
        with tf.variable_scope('conv5/conv5_1'):
            net = conv_layer(net, [3,3,512,512], 512, 'conv5/conv5_1')
            net = tf.nn.relu(net)
        with tf.variable_scope('conv5/conv5_2'):
            net = conv_layer(net, [3,3,512,512], 512, 'conv5/conv5_2')
            net = tf.nn.relu(net)
        with tf.variable_scope('conv5/conv5_3'):
            net = conv_layer(net, [3,3,512,512], 512, 'conv5/conv5_3')
            net = tf.nn.relu(net)  
            net = maxpooling_2x2(net, 'pool5')  
            
        with tf.variable_scope('conv6'):
            net = conv_layer(net, [1,1,512,1028], 1028, 'conv6') #receptive field = 30
            net = tf.nn.relu(net)               
            
        with tf.variable_scope('conv7'):
            net = conv_layer(net, [1,1,1028,1028], 1028, 'conv7')#receptive field = 28]
            net = tf.nn.relu(net)
                
        with tf.variable_scope('conv8'):
            net = conv_layer(net, [1,1,1028,256], 256, 'conv8')#receptive field = 28]
                #net = tf.nn.relu(net)
                
        with tf.variable_scope('deconv_1'): 
            net = deconv_layer(net,[2,2,256,256],[batch_size,32,32,256], name = 'deconv_1')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_2'): 
            net = deconv_layer(net,[2,2,256,256],[batch_size,64,64,256], name = 'deconv_2')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_3'): 
            net = deconv_layer(net,[2,2,256,256],[batch_size,128,128,256], name = 'deconv_3')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_4'): 
            net = deconv_layer(net,[2,2,256,256],[batch_size,256,256,256], name = 'deconv_4')
            net = tf.nn.relu(net)
            
        with tf.variable_scope('deconv_5'): 
            net = deconv_layer(net,[2,2,256,256],[batch_size,512,512,256], name = 'deconv_5')
            net = tf.nn.relu(net)   
        with tf.variable_scope('logit'):
            net = conv_layer(net, [1,1,256,class_num], class_num, 'logit')#receptive field = 28]
                #net = tf.nn.relu(net)   
            
    return net    

def vgg16(inputs, batch_size, class_num, reuse=False,
             is_training=True, scope='vgg_16'):
    
    with tf.variable_scope(scope, 'vgg_16',[input], reuse=reuse):
            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            
            with tf.variable_scope('conv6'):
                net = conv_layer(net, [1,1,512,1028], 1028, 'conv6') #receptive field = 30
                net = tf.nn.relu(net)               
            
            with tf.variable_scope('conv7'):
                net = conv_layer(net, [1,1,1028,1028], 1028, 'conv7')#receptive field = 28]
                net = tf.nn.relu(net)
                
            with tf.variable_scope('conv8'):
                net = conv_layer(net, [1,1,1028,256], 256, 'conv8')#receptive field = 28]
                #net = tf.nn.relu(net)
                
            with tf.variable_scope('deconv_1'): 
                net = deconv_layer(net,[2,2,256,256],[batch_size,32,32,256], name = 'deconv_1')
                net = tf.nn.relu(net)
            
            with tf.variable_scope('deconv_2'): 
                net = deconv_layer(net,[2,2,256,256],[batch_size,64,64,256], name = 'deconv_2')
                net = tf.nn.relu(net)
            
            with tf.variable_scope('deconv_3'): 
                net = deconv_layer(net,[2,2,256,256],[batch_size,128,128,256], name = 'deconv_3')
                net = tf.nn.relu(net)
            
            with tf.variable_scope('deconv_4'): 
                net = deconv_layer(net,[2,2,256,256],[batch_size,256,256,256], name = 'deconv_4')
                net = tf.nn.relu(net)
            
            with tf.variable_scope('deconv_5'): 
                net = deconv_layer(net,[2,2,256,256],[batch_size,512,512,256], name = 'deconv_5')
                net = tf.nn.relu(net)   
            with tf.variable_scope('logit'):
                net = conv_layer(net, [1,1,256,class_num], class_num, 'logit')#receptive field = 28]
                #net = tf.nn.relu(net)    
    return net    


                                       
    
    
    
    
    
    
    