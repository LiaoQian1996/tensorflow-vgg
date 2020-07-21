"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import os
import numpy as np
import collections
import vgg19_trainable as vgg19
import utils

batch_size = 2
crop_size = 224
path = 'D:/WorkSpace/DATA/marble_test_png/'
true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

image_list = os.listdir(path)    
print(image_list)
image_list = [_ for _ in image_list if _.endswith('.png')]
if len(image_list)==0:
    raise Exception('No png files in the input directory !')
image_list = [os.path.join(path, _) for _ in image_list]

with tf.variable_scope('load_image'):
        filename_queue = tf.train.slice_input_producer([image_list],
                                               shuffle=True, capacity=128)
        print('filename_queue : ',filename_queue)
        reader = tf.WholeFileReader()
        value = tf.read_file(filename_queue[0])
        image = tf.image.decode_png(value, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(image)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            image = tf.identity(image)
            
with tf.name_scope('crop'):
    image = tf.image.crop_to_bounding_box(image, 0, 0, crop_size, crop_size)
    
image_batch, filename_batch = tf.train.batch([image, filename_queue],\
                                        batch_size = batch_size,\
                                        capacity = 128,\
                                        num_threads = 4) 

with tf.device('/gpu:0'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    train_mode = tf.placeholder(tf.bool)
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
#     vgg = vgg19.Vgg19('./vgg19.npy')

    true_out = tf.placeholder(tf.float32, [1, 1000])
    
    vgg = vgg19.Vgg19()
    vgg.build(image_batch, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # test classification
#     prob = sess.run(vgg.prob, feed_dict={train_mode: False})
#     utils.print_prob(prob[0], './synset.txt')

    # simple 1-step training
#     cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = true_out, logits=vgg.logits))
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    for i in range(10):
        cost_, _ = sess.run([cost, train], feed_dict={true_out: [true_result], train_mode: True})
        if i%2 == 0:
            print('Iteration : %i'%i)
            print('cost : %f'%cost_)
            
#     # test classification again, should have a higher probability about tiger
#     prob = sess.run(vgg.prob, feed_dict={train_mode: False})
#     utils.print_prob(prob[0], './synset.txt')

#     # test save
#     vgg.save_npy(sess, './test-save.npy')
