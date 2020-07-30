"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import os
import numpy as np
import collections
import vgg19_trainable as vgg19
import utils
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class_num = 130
crop_size = 224

batch_size = 1
max_iteration = 20000
save_step = max_iteration
display_step = 20
summary_step = 100
initial_learning_rate = 0.0001
# path = 'F:/marble130_dataset/test/'
log_dir = './log/'
path = './my_test_data/'
# path = '/media/liaoqian/Seagate2/marble130_dataset/train/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
for _ in os.listdir(log_dir):
    os.remove(log_dir + _)
with tf.device('/cpu:0'):
    with tf.variable_scope('load_image'):
            filename_list = os.listdir(path)    # path内应只有.png图像文件
            label_list = [int(filename.split('_')[0])-1 for filename in filename_list]
            if len(filename_list)==0:
                raise Exception('No png files in the input directory !')
            filename_list = [os.path.join(path, _) for _ in filename_list]
            queue = tf.train.slice_input_producer([filename_list,label_list], \
                                                           shuffle=True, capacity=128)
            label = queue[1]
            image = tf.read_file(queue[0])
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32) # [0,255] -> [0., 1.]
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

    image_batch, label_batch = tf.train.batch([image, label],\
                                            batch_size = batch_size,\
                                            capacity = 128,\
                                            num_threads = 4) 

with tf.name_scope('build_vgg_model_and_compute_graph'):  
#     vgg = vgg19.Vgg19('./20200727.npy')
    vgg = vgg19.Vgg19()
    label_batch = tf.one_hot(label_batch, class_num)
    print('label_batch : ', label_batch.shape)
    train_mode = tf.placeholder(tf.bool)
    vgg.build(image_batch, train_mode)
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_batch, logits=vgg.logits))
    
with tf.name_scope('learning_rate_decay'): 
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10,
                                               decay_rate=0.5)

    opt = tf.train.GradientDescentOptimizer(learning_rate)
    add_global_step = global_step.assign_add(1)
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('summary_info'):
    tf.summary.scalar('cross_entropy_cost',cost)
#     tf.summary.image('input_image',tf.image.convert_image_dtype(image_batch,\
#                                                             dtype=tf.uint8, saturate=True))


with tf.device('/gpu:0'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)  

    for i in range(max_iteration):
        fetches = {
            "train" : train,
            "global_step" : add_global_step
        }
        
        if i%display_step == 0:
            fetches["cost"] = cost
            
        if i%summary_step == 0:
            fetches["class_num_pred"] = tf.argmax(vgg.logits, axis = 1)
            fetches["summary"] = merged
            
        results = sess.run(fetches, feed_dict = {train_mode: True})
        
        if i%display_step == 0:
            print('Iteration : %i'%i)
            print('cost : %f'%results["cost"])
        if i%summary_step == 0:
            train_writer.add_summary(results["summary"], i+1)
            print('当前batch的标签为 ： ', class_num_batch)
            print('当前batch的预测为 ： ', results["class_num_pred"])

#     # test save
    if i == max_iteration:
        vgg.save_npy(sess, './base_on_20200727.npy')
