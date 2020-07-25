"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import os
import numpy as np
import collections
import vgg19_trainable as vgg19
import utils

batch_size = 5
crop_size = 224
class_num = 130
max_iteration = 100000
# path = 'D:/WorkSpace/DATA/marble_test_png/'
# path = '/home/liaoqian/DATA/data_609/'
path = 'F:/marble130_dataset/test/'


with tf.variable_scope('load_image'):
        image_list = os.listdir(path)    
        image_list = [_ for _ in image_list if _.endswith('.png')]
        if len(image_list)==0:
            raise Exception('No png files in the input directory !')
        image_list = [os.path.join(path, _) for _ in image_list]
        filename_queue = tf.train.slice_input_producer([image_list], \
                                                       shuffle=True, capacity=128)
        reader = tf.WholeFileReader()
        value = tf.read_file(filename_queue[0])
        image = tf.image.decode_png(value, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(image)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            image = tf.identity(image)

with tf.name_scope('random_crop'):
    print('[Config] Use random crop')   
    input_size = tf.shape(image)
    h, w = tf.cast(input_size[0], tf.float32),\
    tf.cast(input_size[1], tf.float32)
    offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, w - crop_size)),
                       dtype=tf.int32)
    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, h - crop_size)),
                       dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_size, crop_size)  
    
    
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

    true_out = tf.placeholder(tf.float32, [batch_size, 130])
    
    vgg = vgg19.Vgg19()
    vgg.build(image_batch, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = true_out, logits=vgg.logits))
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    for i in range(max_iteration):
        filename_batch_ = sess.run(filename_batch)
        filename_batch_ = [_[0].decode() for _ in filename_batch_]
        # 从而得到一个batch的文件名
        # print(filename_batch_)
        
        # 下面从一个batch的文件名中提取出一个batch的one-hot编码的label，根据文件名的组织形式修改之
        class_num_batch = [int(_.split('/')[-1].split('_')[0])-1 for _ in filename_batch_]
        print('Class of this image is ', class_num_batch)
        # label_batch 即为这个batch图像对应的类别标签
        label_batch = np.eye(class_num)[class_num_batch]
        print(label_batch.shape)
        cost_, _ = sess.run([cost, train], feed_dict={true_out: label_batch, train_mode: True})
        if i%100 == 0:
            print('Iteration : %i'%i)
            print('cost : %f'%cost_)
        if i%1000 == 0:
            with open("cost_record.txt","w") as f:
                f.write("iteration : %i    cost : .4f \n"%(i, cost_))

#     # test save
    vgg.save_npy(sess, './20200725.npy')
