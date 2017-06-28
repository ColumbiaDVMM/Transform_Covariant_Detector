#!/bin/bash

datasets=("VggAffineDataset" "EFDataset" "WebcamDataset")
#datasets=("VggAffineDataset")
train_name="mexico_patch_train_2d_new_iter_5"
stats_name="mexico_patch_train_2d_new"

for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    python patch_network_point_test.py --train_name $train_name --stats_name $stats_name --dataset_name $dataset
done
