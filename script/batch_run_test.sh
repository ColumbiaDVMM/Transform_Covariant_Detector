#!/bin/bash

cd ..

dataset_name='NC2017_Dev2_Beta1'
conv_feature_name='covariant_point_2d_tilde'
feature_name='feature_point_2d_tilde'
network_name='mexico_patch_train_2d_new_gp_2d_iter_15'
stats_name='mexico_patch_train_2d'

point_number='500'
matlab="/home/xuzhang/MATLAB/bin/matlab -c /home/xuzhang/tool/Mathworks_Matlab_R2015a_Linux/fixr2015arel/Standalone.lic -nodisplay"

python patch_network_point_test.py --train_name $network_name --stats_name $stats_name --dataset_name $dataset_name --save_feature $conv_feature_name

cd ../test/

$matlab -r "point_extractor_2d('$dataset_name','$conv_feature_name','$feature_name',$point_number);  exit(0);";
