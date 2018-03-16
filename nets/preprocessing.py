from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def data_augmentation(image, is_training=None):
    
    if is_training:
        return preprocess_for_train(image)
    
    else:
        return preprocess_for_eval(image)
    
   
    
def preprocess_for_train(image):
    #distorted_image = tf.image.random_brightness(image, max_delta=63)!!!Is nothing after random_brightness
    distorted_image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return distorted_image
def preprocess_for_eval(image):
    #distorted_image = tf.image.random_brightness(image, max_delta=63)!!!Is nothing after random_brightness
    distorted_image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return distorted_image
    