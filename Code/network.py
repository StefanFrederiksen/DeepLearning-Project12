#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:11:18 2017

@author: jacob
"""

import numpy as np
import tensorflow as tf

import utils 
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.contrib.layers import flatten

path = ''
show_dimensions = True

load_model = True
save_model = True
batch_size = 32
max_epochs = 10
valid_every = 100

"""
Data loader
"""

tf.reset_default_graph()

num_classes = 10
height, width, nchannels = 128, 128, 1
padding = 'same'

filters_1 = 1
kernel_size_1 = (1,1) 
pool_size_1 = (1,1)

x_pl = tf.placeholder(tf.float32, [None, height, width, nchannels], name='xPlaceholder')
y_pl = tf.placeholder(tf.float32, [None, num_classes], name='yPlaceholder')

"""
The network
"""
with tf.variable_scope('convlayer1'):
    conv1 = Conv2D(filters_1, kernel_size_1, strides=(2,2), padding=padding, activation='relu')
    x = conv1(x_pl)   
    pool1 = MaxPooling2D(pool_size=pool_size_1, strides=None, padding=padding)    
    x = pool1(x)    
    
    if show_dimensions == True:
        print('input\t\t', x_pl.get_shape())
        print('output layer1\t', x.get_shape())
    
with tf.variable_scope('output_layer'):
    denseOut = Dense(units=num_classes, activation='softmax')
    y = denseOut(x)
    
    if show_dimensions == True:
        print('final output\t', y.get_shape())
        
"""
Loss, Optimizer, Accuracy, etc
"""
with tf.variable_scope('loss'):
    #cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), axis=[1])
    #average cross entropy
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.variable_scope('training'):
    #define optimizer
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    #apply it
    train_op = optimizer.minimize(cross_entropy)
    
with tf.variable_scope('performance'):
    #comparing results with labels
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
    
    """
    Implement correct_predicgtion when multi labels present
    """
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Training loop
"""


valid_loss, valid_accuracy = [],[]
train_loss, train_accuracy = [],[]
test_loss, test_accuracy = [],[]
saver = tf.train.Saver()
batches_completed = 0 

with tf.Session() as sess:
    if load_model == True:
        saver.restore(sess, "../model/model.ckpt")
        print("Model restored")
    else:
        sess.run(tf.global_variables_initializer())
    print("\ttrain_loss \ttrain_acc \tvalid_loss \tvalid_acc")
    try:
        while epochs_completed < max_epochs:
            _train_loss, _train_accuracy = [],[]
            
            # Run training
            x_batch, y_batch = "data loader here"
            fetches_train = [train_op, cross_entropy, accuracy]
            feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
            _, _loss, _acc = sess.run(fetches_train, feed_dict_train)
            
            _train_loss.append(_loss)
            _train_accuracy.append(_acc)
            
            # Compute validation, loss and accuracy
            if batches_completed % valid_every == 0:
                train_loss.append(np.mean(_train_loss))
                train_accuracy.append(np.mean(_train_accuracy))
                
                fetches_valid = [cross_entropy, accuracy]
                
                feed_dict_valid = {x_pl: x_valid, y_pl: y_valid}
                _loss, _acc = sess.run(fetches_valid, feed_dict_valid)
                
                valid_loss.append(_loss)
                valid_accuracy.append(_acc)
                
                print("%d:\t  %.2f\t\t  %.1f\t\t  %.2f\t\t  %.1f" \
                      % (batches_completed, train_loss[-1], train_accuracy[-1], \
                         valid_loss[-1], valid_accuracy[-1]))
                
                
    except KeyboardInterrupt:
        pass
    
    
    
    
    if save_model == True:
        save_path = (sess, "../model/model.ckpt")
        print("Model saved in file: %s" % save_path)