##Learning Discriminative and Transformation Covariant Local Feature Detectors

This code is the training and evaluation code for our CVPR 2017 paper. It includes the implement of a translation covariant local feature detector. The affine covariant model will be added in the future. 

@inproceedings{zhang2017learning,
  title={Learning Discriminative and Transformation Covariant Local Feature Detectors},
  author={Zhang, Xu and Yu, Felix X. and Karaman, Svebor and Chang, Shih-Fu},
  booktitle={CVPR},
  year={2017}
}

The code is tested on Ubuntu 14.04

###Requirement
Python package:

tensorflow>1.0.0, tqdm, cv2, exifread, skimage, glob

###Usage
####Get the data
Download data from 
https://www.dropbox.com/s/l7a8zvni6ia5f9g/datasets.tar.gz?dl=0

and put the extract the data to ./data/

####Run the code
cd ./script

Generate transformed patch and train the model

./batch\_run_train.sh

Extract local feature point

./batch\_run_test.sh

Evaluate the performance

./batch\_run_eval.sh

###Acknowledgement

We would like to thank

VLfeat [1], http://www.vlfeat.org/ 

Tilde [2], https://github.com/kmyid/TILDE

Karel Lenc etal [3], https://github.com/lenck/ddet

for offering the implementations of their methods. 

and

Vgg dataset [3]

EF dataset [5]

Webcam dataset [2]

for providing the image data.

[1] A. Vedaldi and B. Fulkerson, VLFeat: An Open and Portable Library of Computer Vision Algorithms

[2] Y. Verdie, K. M. Yi, P. Fua, and V. Lepetit. Tilde: A temporally invariant learned detector. CVPR 2015

[3] K. Lenc and A. Vedaldi. Learning covariant feature detectors. In ECCV Workshop on Geometry Meets Deep Learning,2016.

[4] K. Mikolajczyk, T. Tuytelaars, C. Schmid, A. Zisserman, J. Matas, F. Schaffalitzky, T. Kadir and L. Van Gool, A comparison of affine region detectors. IJCV 2005.

[5] C. L. Zitnick, K. Ramnath, Edge foci interest points, ICCV, 2011