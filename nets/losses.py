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

"""Definition of 512 SpinePathNet losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from collections import namedtuple
import tensorflow as tf
import numpy as np


def collect_random_points_batch(mask_instance, batch_size, selected_points_num):
    
    #First, compute the number of instances in a batch.
    instance_number = tf.reduce_max(mask_instance)    
    i = tf.constant(0)    
    #random_index_all_instances = tf.constant(tf.ones([1, 3],dtype =tf.int32))  
    random_index_all_instances = tf.constant([[0,1,1]],dtype =tf.int32) 
    while_condition = lambda i, random_index_all_instances: tf.less_equal(i, tf.to_int32(instance_number))
    def body(i, random_index_all_instances):
        #Second, find all points of one instance in a batch.
        #one_instance_points = tf.where(tf.equal(tf.to_int32(mask_instance[:,:,:,0]),i))
        one_instance_points = tf.where(tf.equal(tf.to_int32(mask_instance),i))
        #Third, find the number of points of one instance in a batch.
        points_number = tf.div(tf.size(one_instance_points),3)
        
        #Random generate 10*batch number of one instance in a batch.
        selected_points_one_batch = selected_points_num*batch_size
        instance_random_number = tf.random_uniform([selected_points_one_batch,1], maxval=points_number, dtype=tf.int32)
    
        # Random select 10*batch points of one instance in a batch.
        random_index_one_instance = tf.gather_nd(one_instance_points,instance_random_number)
        random_index_all_instances = tf.concat([random_index_all_instances, tf.to_int32(random_index_one_instance)],axis=0)
        return [i+1, random_index_all_instances]

    index, random_index = tf.while_loop(while_condition, body, [i, random_index_all_instances],
                                        shape_invariants=[i.get_shape(), tf.TensorShape([None, 3])])
    return (tf.to_int32(instance_number)+1)*selected_points_num*batch_size, random_index  ### +1 means adding background instance.

def similarity_two_points(i,j):
    #Squared Euclidean distance between two vectors:
    #sess = tf.Session()
    distance = tf.reduce_sum(tf.sqrt(tf.subtract(i, j)), name ='Absoluted_Euclidean_distance')
    ###Computer similarity s between pixels p and q as follows:
    similarity = tf.div(2., tf.add(tf.exp(distance), 1.), name='computer_similarity')
    return tf.to_float(similarity)

##Difine log_loss according to two points similarity.
def log_loss(similarity,x,y):
    #z1 = -tf.log(similarity)
    #z2 = -tf.log(tf.subtract(1.,similarity))
    z1 = -tf.log(similarity + tf.constant(1e-08))
    z2 = -tf.log(tf.subtract(1.,similarity) + tf.constant(1e-08))#### avoiding infinate loss.
    log_loss = tf.cond(tf.equal(x,y),lambda:z1,lambda:z2)

    return log_loss


def i_with_others_loss(i, indexs_num, instance_label_i, feature_i, mask_instance,indexs,feature_embedding):
    j = tf.to_int32(i + 1) 
    loss_j = tf.constant(0,dtype=tf.float32)
    while_condition_j = lambda j, loss_j: tf.less(j, tf.to_int32(indexs_num))
    def body_j(j,loss_j):
        instance_label_j = tf.gather_nd(mask_instance[:,:,:,0],indexs[j])
        feature_j = tf.gather_nd(feature_embedding,[indexs[j]])
        similarity = similarity_two_points(feature_i,feature_j)
        loss = log_loss(similarity, instance_label_i, instance_label_j)
        loss_j = tf.add(tf.to_float(loss_j), tf.to_float(loss))        
        return [tf.add(j,1), loss_j]
    j_num, loss_j = tf.while_loop(while_condition_j, body_j, [j, loss_j])
    
    return tf.div(tf.to_float(loss_j), tf.to_float(j_num+1))
    

def instance_loss(FLAGS,feature_embedding, mask_instance):

    ###resize mask to be same as the feture_embedding.
    #loss_batch_norm = tf.to_float(tf.gather_nd(feature_embedding,[[1,1,1,0]]))
    batch_size = FLAGS.batch_size
    #size = (feature_height, feature_width)
    #mask_instance = tf.image.resize_nearest_neighbor(mask_instance, (128,128))
    #feature_embedding = tf.image.resize_bilinear(feature_embedding, (128,128))
    print('Construting loss graph, which needs much time, please wait for a moment.') 
    selected_points_num = 2
    indexs_num, indexs = collect_random_points_batch(mask_instance, batch_size,selected_points_num)    
    i = tf.constant(0,dtype=tf.int32)
    loss_i = tf.constant(0,dtype=tf.float32)
    while_condition_i = lambda i, loss_i: tf.less(i, tf.to_int32(indexs_num))
    def body_i(i,loss_i):
        instance_label_i = tf.gather_nd(mask_instance,indexs[i])
        feature_i = tf.gather_nd(feature_embedding,[indexs[i]])
        loss = i_with_others_loss(i, indexs_num, instance_label_i, feature_i, mask_instance,indexs,feature_embedding)
        loss_i = tf.add(tf.to_float(loss_i), tf.to_float(loss))
        return [tf.add(i,1), loss_i]
    i_num, loss_i = tf.while_loop(while_condition_i, body_i, [i, loss_i],
                                  shape_invariants=[i.get_shape(), tf.TensorShape([])])
    return tf.div(loss_i,tf.to_float(i_num)+1.)


def metric_loss(FLAGS,feature_embedding, mask_class):
    
    """Enlarge different classes feature space distance and reduce the same classes distance. 
    Input:
    feature_embedding is the last feature layers [batch_size, hight, width, channels]
    mask_class is the class labels [batch_size, hight, channels]    
    Output:
    The triplet loss.    
    """ 
    batch_size = FLAGS.batch_size
    selected_points_num = 5 # selected points_num of each class.
    indexs_num, indexs = collect_random_points_batch(mask_class, batch_size, selected_points_num)
    i = tf.constant(0, dtype=tf.int32)
    loss_i = tf.constant(0,dtype=tf.float32)
    while_condition_i = lambda i, loss_i: tf.less(i, tf.to_int32(indexs_num))
    def body_i(i,loss_i):
        class_label_i = tf.gather_nd(mask_class,indexs[i])
        feature_i = tf.gather_nd(feature_embedding,[indexs[i]])
        loss = one_pixel_class_loss(i, indexs_num, class_label_i, feature_i, mask_class,indexs,feature_embedding) 
        #tf.add_to_collection('loss_i',loss)
        #loss_i = tf.add_n(tf.get_collection('loss_i'), name='one_class_loss')
        loss_i = tf.add(tf.to_float(loss_i), tf.to_float(loss))
        return [tf.add(i,1), loss_i]    
    i_num, loss_i = tf.while_loop(while_condition_i, body_i, [i, loss_i],
                                  shape_invariants=[i.get_shape(), tf.TensorShape([])])
    class_loss = tf.div(loss_i,tf.to_float(i_num)+1.)
    tf.add_to_collection('class_loss', class_loss)
    class_losses = tf.add_n(tf.get_collection('class_loss'), name='total_class_loss')
    #tf.losses.add_loss(class_losses)
    return class_losses

def one_pixel_class_loss(i, indexs_num, class_label_i, feature_i, mask_class,indexs,feature_embedding):
    j = tf.to_int32(i + 1) 
    loss_j = tf.constant(0,dtype=tf.float32)
    while_condition_j = lambda j, loss_j: tf.less(j, tf.to_int32(indexs_num))
    def body_j(j,loss_j):
        class_label_j = tf.gather_nd(mask_class,indexs[j])
        feature_j = tf.gather_nd(feature_embedding,[indexs[j]])
        similarity = similarity_two_points(feature_i,feature_j)
        loss = log_loss(similarity, class_label_i, class_label_j)
        #tf.add_to_collection('loss_j',loss)
        #loss_j = tf.add_n(tf.get_collection('loss_j'), name='one_pixel_loss')
        loss_j = tf.add(tf.to_float(loss_j), tf.to_float(loss))        
        return [tf.add(j,1), loss_j]
    j_num, loss_j = tf.while_loop(while_condition_j, body_j, [j, loss_j])    
    return tf.div(tf.to_float(loss_j), tf.to_float(j_num+1))


def weighted_cross_entropy_with_logits(FLAGS, Fold, logits, labels):
    
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          
      labels: Labels tensor, int32 - [-1, num_classes].
          The ground truth of your data.
      weights: lists - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    num_classes = FLAGS.num_classes
    #for logits
    logits = tf.reshape(logits, (-1, num_classes))        
    epsilon = tf.constant(value=1e-08)    
    softmax = tf.nn.softmax(logits) + epsilon
    
    
    """
    weights is calculated in training sets by check_tfrecords.
    SPINE_LABELS = {
    'none':(0,'background'),
    'vn':(1, 'Normal Vertebral'),
    'vd':(2, 'Vertebral Deformity'),
    'dn':(3, 'Normal Disc'),  
    'dm':(4, 'Mild Gegeneration Disc'),
    'ds':(4, 'Severe Degeneration Disc'),
    'fn':(5, 'Neuro Foraminal Normal'),
    'fs':(6, 'Neuro Foraminal Stenosis'),
    'sv':(0, 'Caudal Vertebra')}
    """
    # For different folds according to the dataset of 2017-07-28 version.
    weights_1 = tf.convert_to_tensor([1.1254597, 18.403364, 40.579597, 113.45259, 77.841278, 174.17633, 196.42009])
    
    weights_2 = tf.convert_to_tensor([1.128598, 18.368553, 38.48455, 111.13625, 75.274231, 165.86247, 192.02112])
    
    weights_3 = tf.convert_to_tensor([1.1276629, 17.874931, 41.686779, 110.83454, 77.200882, 164.07047, 192.05385])
    
    weights_4 = tf.convert_to_tensor([1.1253909, 18.315485, 40.971313, 118.71, 77.004868, 171.51686, 193.2919])
    
    weights_5 = tf.convert_to_tensor([1.1280274, 17.963346, 40.372585, 108.81972, 79.39196, 160.1044, 198.91023])  
    
    weights_all = [weights_1,weights_2,weights_3,weights_4,weights_5]
    
    #weights = weights_'%d' %Fold
    weights = weights_all[Fold-1]
    
    if weights is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                           weights), reduction_indices=[1])
    else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')

    tf.add_to_collection('pixel_wise_losses', cross_entropy_mean)
    
    loss_logits = tf.add_n(tf.get_collection('pixel_wise_losses'), name='total_pixel_loss')
    
    return loss_logits


def cross_entropy_with_logits(FLAGS,logits,mask_class):
    #logits = tf.reshape(logits, (-1, FLAGS.num_classes))
    #mask_class = tf.reshape(mask_class, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = mask_class, logits = logits, name='Cross_Entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('pixel_wise_loss', cross_entropy_mean)
    loss_logits = tf.add_n(tf.get_collection('pixel_wise_loss'), name='total_pixel_loss')
    return loss_logits