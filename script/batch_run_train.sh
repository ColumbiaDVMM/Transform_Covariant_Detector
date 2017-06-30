#!/bin/bash
cd ../tensorflow/

matlab="/home/xuzhang/MATLAB/bin/matlab -c /home/xuzhang/tool/Mathworks_Matlab_R2015a_Linux/fixr2015arel/Standalone.lic -nodisplay"
$matlab -r "get_training_pair;  exit(0);";

python patch_network_train_point.py --training mexico_tilde_p24_Mexico_train_point --test mexico_tilde_p24_Mexico_test_point | tee ../log/training_log.txt
