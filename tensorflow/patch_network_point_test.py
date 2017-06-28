from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import gc
import patch_reader
import patch_cnn_point_test
import numpy as np
from scipy.spatial import distance
import scipy.io as sio
import glob
import pickle
import cv2
import os
from skimage.transform import pyramid_gaussian

import argparse

def read_image_from_name(file_name):
    """Return the factorial of n, an exact integer >= 0. Image rescaled to no larger than 1024*768
    
    Args:
        root_dir (str): Image directory
        filename (str): Name of the file

    Returns:
        np array: Image 
        float: Rescale ratio

    """
    img = cv2.imread(file_name)
    if img.shape[2] == 4 :
        img = img[:,:,:3]
        
    if img.shape[2] == 1 :
            img = np.repeat(img, 3, axis = 2)
    #if img.shape[2] == 3 :
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ftest = open(file_name, 'rb')
    tags = exifread.process_file(ftest)
    
    try:
        #print(tags['Thumbnail Orientation'])
        if str(tags['Thumbnail Orientation']) == 'Rotated 90 CW':
            img = cv2.transpose(img)  
            img = cv2.flip(img, 1)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 90 CCW':
            img = cv2.transpose(img)  
            img = cv2.flip(img, 0)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 180':
            img = cv2.flip(img, -1)
    except:
        tags = tags

    ratio = 1.0
    if img.shape[0]*img.shape[1]>1024*768:
        ratio = (1024*768/float(img.shape[0]*img.shape[1]))**(0.5)
        img = cv2.resize(img,(int(img.shape[1]*ratio), int(img.shape[0]*ratio)),interpolation = cv2.INTER_CUBIC);
        
    return img, ratio

parser.add_argument("--train_name", nargs='?', type=str, default = 'mexico_patch_train_2d_gp_2d_iter_5',
                    help="Training dataset name")

parser.add_argument("--stats_name", nargs='?', type=str, default = 'mexico_patch_train_2d',
                    help="Training dataset name")

parser.add_argument("--dataset_name", nargs='?', type=str, default = 'VggAffineDataset',
                    help="Training dataset name")

parser.add_argument("--save_feature", nargs='?', type=str, default = 'covariant_point_2d',
                    help="Training dataset name")

parser.add_argument("--alpha", nargs='?', type=float, default = 1.0,
                    help="alpha")

parser.add_argument("--descriptor_dim", nargs='?', type=int, default = 2,
                    help="Number of embedding dimemsion")args = parser.parse_args()

parser = argparse.ArgumentParser()

train_name = args.train_name
stats_name = args.stats_name
dataset_name = args.dataset_name
save_feature_name = args.save_feature 

# Parameters
patch_size = 32
batch_size = 128
descriptor_dim = args.descriptor_dim

print('Loading training stats:')
file = open('../tensorflow_model/stats_%s.pkl'%stats_name, 'r')
mean, std = pickle.load(file)
print(mean)
print(std)

CNNConfig = {
    "patch_size": patch_size,
    "descriptor_dim" : descriptor_dim,
    "batch_size" : batch_size,
    "alpha" : args.alpha,
    "train_flag" : True
}

cnn_model = patch_cnn_point.PatchCNN(CNNConfig)

#dataset information
subsets = []
if dataset_name=='VggAffineDataset' :
    subsets = ['bikes', 'trees', 'graf', 'wall', 'boat', 'bark', 'leuven', 'ubc']

if dataset_name=='EFDataset' :
    subsets = ['notredame','obama','paintedladies','rushmore','yosemite']

if dataset_name=='WebcamDataset' :
    subsets = ['Chamonix','Courbevoie','Frankfurt','Panorama','StLouis']#'Mexico', #not using mexico for test, 

working_dir = '../data/'
load_dir = working_dir + 'datasets/' + dataset_name
save_dir = working_dir + save_feature_name + '/' + dataset_name

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    try:
        saver.restore(sess, "../tensorflow_model/"+train_name+"_model.ckpt")
        print("Model restored.")
    except:
        print('No model found')
        exit()

    for i,subset in enumerate(subsets) :
        index = 1
        if not os.path.exists(save_dir + '/' + subset) :
            os.mkdir(save_dir + '/' + subset)
        for file in glob.glob(load_dir + '/' + subset + '/*.*') :
            file = os.path.basename(file)
            output_list = []
            if file.endswith(".ppm") or file.endswith(".pgm") or file.endswith(".png") or file.endswith(".jpg") :
                image_name = load_dir +  '/' + subset + '/' + file
                print(image_name)
                save_file = file[0:-4] + '.mat'
                save_name =  save_dir + '/' + subset + '/' + save_file
                
                #read image
                img, ratio = read_image_from_name(image_name)
                if img.shape[2] == 1 :
                    img = np.repeat(img, 3, axis = 2)

                #build image pyramid
                pyramid = pyramid_gaussian(img, max_layer = 4, downscale=np.sqrt(2))
                
                #predict transformation
                for (j, resized) in enumerate(pyramid) :
                    fetch = {
                        "o1": cnn_model.o1
                    }

                    resized = np.asarray(resized)
                    resized = (resized-mean)/std
                    resized = resized.reshape((1,resized.shape[0],resized.shape[1],resized.shape[2]))

                    result = sess.run(fetch, feed_dict={cnn_model.patch: resized})
                    result_mat = result["o1"].reshape((result["o1"].shape[1],result["o1"].shape[2],result["o1"].shape[3]))
                    output_list.append(result_mat)

                sio.savemat(save_name,{'output_list':output_list})
