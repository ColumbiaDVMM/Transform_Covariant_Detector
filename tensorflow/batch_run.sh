#!/bin/bash

dataset=(hes sift)

for trainset in "${dataset[@]}"
do
    echo "${trainset}"
    python patch_network_train_point_iter.py --training mexico_patch_train_2d_${trainset} --test mexico_patch_test_2d_${trainset} | tee ../log/${trainset}.txt
done
