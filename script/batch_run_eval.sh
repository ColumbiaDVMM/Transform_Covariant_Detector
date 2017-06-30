#!/bin/bash
cd ../eval/

dataset_name='EFDataset'
num_keypoint=200
matlab="/home/xuzhang/MATLAB/bin/matlab -c /home/xuzhang/tool/Mathworks_Matlab_R2015a_Linux/fixr2015arel/Standalone.lic -nodisplay"

$matlab -r "evaluate_OxfordEFDataset('$dataset_name', $num_keypoint); exit(0)";
