#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:11:18 2017

@author: jacob
"""

import numpy as np
import tensorflow as tf
import os
#import utils 
from tensorflow.contrib.layers import flatten, max_pool2d, conv2d, fully_connected 
from BatchLoader import BatchLoader

path = ''
show_dimensions = True

load_model = False
save_model = True
# todo: model navn
regulazation = True; reg_scale = 0.0001
dropout = True; keep_chance = 0.5

batch_size = 32
max_epochs = 10
valid_every = 100
seed = 1
GPU_FRAC = 0.5


tf.reset_default_graph()

num_classes = 10
height, width, nchannels = 128, 128, 1
padding = 'same'


"""
Parameters for network
"""

filters_1 = 40
kernel_size_1 = (4,4) 
stride_kernel_1 = (2,2)
pool_size_1 = (4,4)
stride_pool_1 = (2,2)

filters_2 = 40
kernel_size_2 = (4,4)
stride_kernel_2 = (2,2)
pool_size_2 = (4,4)
stride_pool_2 = (2,2)

units1 = 1024

Batch = BatchLoader('../Spectrograms', [10], batch_size=batch_size, 
                    num_classes=num_classes, num_features=height, seed=seed)

x_pl = tf.placeholder(tf.float32, [None, height, width, nchannels], name='xPlaceholder')
y_pl = tf.placeholder(tf.float32, [None, num_classes], name='yPlaceholder')

"""
The network
"""
with tf.variable_scope('convlayer1'):
    x = conv2d(x_pl, filters_1, kernel_size_1, stride=stride_kernel_1, padding=padding, activation_fn=tf.nn.relu)   
    x = max_pool2d(x, pool_size_1, stride=stride_pool_1, padding=padding)    
    
    if show_dimensions == True:
        print()
        print('input\t\t\t', x_pl.get_shape())
        print('output convlayer1\t', x.get_shape())
        
with tf.variable_scope('convlayer2'):
    x = conv2d(x, filters_2, kernel_size_2, stride=stride_kernel_2, padding=padding, activation_fn=tf.nn.relu)
    x = max_pool2d(x, pool_size_2, stride=stride_pool_2, padding=padding)
    
    if show_dimensions == True:
        print('output convlayer2\t', x.get_shape())
    
with tf.variable_scope('denselayer1'):
    x = flatten(x)
    x = fully_connected(x, units1, activation_fn=tf.nn.relu)
    
    if dropout == True:
        x = tf.layers.dropout(x, rate=keep_chance)
    if show_dimensions == True:
        print('output denselayer1\t', x.get_shape())
        
with tf.variable_scope('output_layer'):
    y = fully_connected(x,  num_classes, activation_fn=tf.nn.softmax)
    
    if show_dimensions == True:
        print('final output\t\t', y.get_shape())
        print()
        
"""
Loss, Optimizer, Accuracy, etc
"""
with tf.variable_scope('loss'):
    #cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), axis=[1])
    #average cross entropy
    loss = tf.reduce_mean(cross_entropy)
    
    if regulazation == True:
        regularize = tf.contrib.layers.l2_regularizer(reg_scale)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = sum([regularize(param) for param in params])
        loss += reg_term
        

with tf.variable_scope('training'):
    #define optimizer
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    #apply it
    train_op = optimizer.minimize(loss)
    
with tf.variable_scope('performance'):
    #comparing results with labels
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
    
    """
    todo: Implement correct_prediction when multi labels present
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
epochs_completed = 0


gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_FRAC)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    if load_model == True:
        if os.path.exists('../model/') == False:
            print("No model found, initializing from new\n")
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "../model/model.ckpt")
            print("Model restored\n")
    else:
        sess.run(tf.global_variables_initializer())
        
    # Tensorflow Saver
    saver = tf.train.Saver()
        
    print("\ttrain_loss \ttrain_acc \tvalid_loss \tvalid_acc")
    try:
        while epochs_completed < max_epochs:
            _train_loss, _train_accuracy = [],[]
            _valid_loss, _valid_acc = [],[]
            
            # Run training
            for j, batch in enumerate(Batch.gen_train()):
                x_batch, y_batch = batch['data'], batch['labels']
                fetches_train = [train_op, cross_entropy, accuracy]
                feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
                _, _loss, _acc = sess.run(fetches_train, feed_dict_train)
                _train_loss.append(_loss)
                _train_accuracy.append(_acc)
                
                batches_completed += 1
                epochs_completed = Batch.get_cur_epoch()
                
                # Compute validation, loss and accuracy
                if batches_completed % valid_every == 0:
                    keep_chance_temp = keep_chance
                    keep_chance = 1
                    train_loss.append(np.mean(_train_loss))
                    train_accuracy.append(np.mean(_train_accuracy))
                    
                    fetches_valid = [cross_entropy, accuracy]
                    
                    for batch, i in Batch.gen_valid():
                        x_valid = batch['data']
                        y_valid = batch['labels']
                        feed_dict_valid = {x_pl: x_valid, y_pl: y_valid}
                        _loss, _acc = sess.run(fetches_valid, feed_dict_valid)
                        
                        _valid_loss.append(np.mean(_loss))
                        _valid_acc.append(np.mean(_acc))
                        
                    valid_loss.append(np.mean(_valid_loss))
                    valid_accuracy.append(np.mean(_valid_acc))
                    keep_chance = keep_chance_temp
                    print("%d:\t  %.5f\t\t  %.5f\t\t  %.5f\t\t  %.5f" \
                          % (batches_completed, train_loss[-1], train_accuracy[-1], \
                             valid_loss[-1], valid_accuracy[-1]))
        more_test_data = True
        while more_test_data == True:
            x_batch, y_batch = "todo: test data loader"
            feed_dict_test = {x_pl: x_batch, y_pl: y_batch}
            _loss, _acc = sess.run(fetches_valid, feed_dict_test)
            test_loss.append(_loss)
            test_accuracy.append(_acc)
        print('Test loss {:6.3f}, Test acc {:6.3f}'.format(
                np.mean(test_loss), np.mean(test_accuracy)))
                
    except KeyboardInterrupt:
        pass
    
    
    
    
    if save_model == True:
        save_path = saver.save(sess, "../model/model.ckpt")
        print("Model saved in file: %s" % save_path)