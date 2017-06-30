"""
Train a detector with standard patches and transformed patches.

Usage: python patch_network_train_point.py --training mexico_tilde_p24_Mexico_train_point --test mexico_tilde_p24_Mexico_test_point [Option]

General options:
    --learning_rate Starting learning rate (default 0.01)
    --training      Name of the training dataset
    --test          Name of the test dataset
    --alpha         Trade-off parameter between invertible loss and covariant loss
    --num_epoch     Number of epoch to run
    --descriptor_dim Number of the parameter for transformation (translation 2)
    --batch_size    Batch size 
    --patch_size    Default 32

Output:
    
    Output detector
    
Examples:

    >>> python patch_network_train_point.py --training mexico_tilde_p24_Mexico_train_point --test mexico_tilde_p24_Mexico_test_point 
    
"""

from __future__ import print_function

import tensorflow as tf
import gc
import patch_reader
import patch_cnn
import numpy as np
from scipy.spatial import distance
import scipy.io as sio
import pickle
from tqdm import tqdm
from datetime import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", nargs='?', type=float, default = 0.01,
                    help="learning rate")

parser.add_argument("--training", nargs='?', type=str, default = 'mexico_tilde_p24_Mexico_train_point',
                    help="Training dataset name")

parser.add_argument("--test", nargs='?', type=str, default = 'mexico_tilde_p24_Mexico_test_point',
                    help="Training dataset name")

parser.add_argument("--alpha", nargs='?', type=float, default = 1.0,
                    help="alpha")

parser.add_argument("--num_epoch", nargs='?', type=int, default = 20,
                    help="Number of epoch")

parser.add_argument("--descriptor_dim", nargs='?', type=int, default = 2,
                    help="Number of embedding dimemsion")

parser.add_argument("--batch_size", nargs='?', type=int, default = 128,
                    help="Number of embedding dimemsion")

parser.add_argument("--patch_size", nargs='?', type=int, default = 32,
                    help="Number of embedding dimemsion")

args = parser.parse_args()

# Parameters
start_learning_rate = args.learning_rate
num_epoch = args.num_epoch
display_step = 100
batch_size =args.batch_size
patch_size =args.patch_size
num_epoch = args.num_epoch
now = datetime.now()
suffix = 'LR{:1.0e}_alpha{:1.1e}'.format(start_learning_rate,args.alpha) + now.strftime("%Y%m%d-%H%M%S")
descriptor_dim = args.descriptor_dim

train = patch_reader.SiameseDataSet('../data/patch_set/train_pair/')
train.load_by_name(args.training, patch_size = patch_size)

test = patch_reader.SiameseDataSet('../data/patch_set/train_pair/')
test.load_by_name(args.test, patch_size = patch_size)

print('Loading training stats:')
try:
    file = open('../data/stats_%s.pkl'%args.training, 'r')
    mean, std = pickle.load(file)
except:
    print('No precompute stats! Calculate and save the stats from training data.')
    mean, std = train.generate_stats()
    pickle.dump([mean,std], open('../data/stats_%s.pkl'%args.training,"wb"));
print('-- Mean: %s' % mean)
print('-- Std:  %s' % std)

#normalize data
train.normalize_data(mean, std)
test.normalize_data(mean, std)

# get patches
patches_train = train._get_patches()
patches_train_t = train._get_patches_transformed()

patches_test  = test._get_patches()
patches_test_t = test._get_patches_transformed()

#get gt transform
patches_train_label = train._get_matrix()
patches_test_label = test._get_matrix()

train.generate_index()
test.generate_index()
# get matches for evaluation
print('Learning Rate: {}'.format(args.learning_rate))
print('Feature_Dim: {}'.format(args.descriptor_dim))
print('Alpha: {}'.format(args.alpha))

#set up network
CNNConfig = {
    "patch_size": patch_size,
    "descriptor_dim" : descriptor_dim,
    "batch_size" : batch_size,
    "alpha" : args.alpha,
    "train_flag": True
}
cnn_model = patch_cnn.PatchCNN(CNNConfig)

#time decayed learning rate
global_step = tf.Variable(0, trainable=False)
decay_step  = 1000
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                           decay_step, 0.96, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9).minimize(cnn_model.cost,global_step=global_step)

saver = tf.train.Saver()
training_logs_dir = '../log/tensorflow/train_{}/'.format(suffix)
test_logs_dir = '../log/tensorflow/test_{}/'.format(suffix)

# Initializing the variables
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

# Launch the graph
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    step = 1
    training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_logs_dir, sess.graph)
    merged = tf.summary.merge_all()
    
    for i in range(num_epoch):
        epoch_loss = 0
        num_batch_in_epoch = train.num_train_patch//batch_size
        print('Start iteration {}'.format(i))
        for step in tqdm(xrange(num_batch_in_epoch)):
            #training 
            index = train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={cnn_model.patch: patches_train[index],\
                    cnn_model.patch_t: patches_train_t[index], \
                    cnn_model.label: patches_train_label[index]})

            #Show loss        
            if step>0 and step % display_step == 0:
                step = step+1
                fetch = {
                   "o1" : cnn_model.o1_flat,
                   "o2" : cnn_model.o2_flat,
                   "cost": cnn_model.cost,
                   "inver": cnn_model.inver_loss,
                   "covar_loss": cnn_model.covariance_loss,
                }
               
                summary, result = sess.run([merged,fetch], feed_dict={cnn_model.patch: patches_train[index], \
                     cnn_model.patch_t: patches_train_t[index], \
                     cnn_model.label: patches_train_label[index]})
                
                #print(result["o1"][0:1])
                #print(result["o2"][0:1]) 
                
                training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                epoch_loss = epoch_loss+result["cost"]

                #show test loss
                index = test.next_batch(batch_size)
                fetch = {
                    "cost": cnn_model.cost,
                    "inver": cnn_model.inver_loss,
                    "covar_loss": cnn_model.covariance_loss,
                    "o1" : cnn_model.o1_flat, 
                }
                
                summary, result = sess.run([merged,fetch], feed_dict={cnn_model.patch: patches_test[index], \
                     cnn_model.patch_t: patches_test_t[index], \
                     cnn_model.label: patches_test_label[index]})
                #print('test')
                #print(result["o1"][0:1])
                test_writer.add_summary(summary, tf.train.global_step(sess, global_step))
        #save model
        if i > 0 and (i+1)%5==0:
            model_ckpt = '../tensorflow_model/'+args.training+'_translation_iter_'+ str(i+1) +'_model.ckpt'
            saver.save(sess, model_ckpt)

training_writer.close()
test_writer.close()
