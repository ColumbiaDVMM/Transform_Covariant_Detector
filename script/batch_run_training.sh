#!/bin/bash
cd ..
alpha_set=(1.0)
for alpha in "${alpha_set[@]}"
do
    echo "Alpha: ${alpha}"
    python patch_network_train_point.py --training NC2017_Dev2_Beta1_train_2d_sift --test NC2017_Dev2_Beta1_test_2d_sift --alpha ${alpha} | tee ../log/${alpha}.txt
done
