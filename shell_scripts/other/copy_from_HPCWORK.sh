#!/bin/bash

# find all the final models in the directory and save them in an array

model_dir=$HPCWORK/Mask2Former/CLAIX_OUTPUT
save_to=$HOME/Mask2Former/CLAIX_OUTPUT

models=($(dir $model_dir))

for model in ${models[@]}; do
    echo $model
    # check if the directory exists
    if [ -d $save_to/$model ]; then
        continue
    fi
    mkdir -p $save_to/$model
    rsync $model_dir/$model/model_final.pth $save_to/$model
    rsync $model_dir/$model/config.yaml $save_to/$model
    rsync $model_dir/$model/metrics.json $save_to/$model
    rsync $model_dir/$model/log.txt $save_to/$model
done