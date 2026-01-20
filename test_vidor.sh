#!/bin/sh
gpu_id=0
batch_size=32
val_traj=meta
# val_traj=gt
ckpt_path='your/ckpt/path'
CUDA_VISIBLE_DEVICES=$gpu_id python test.py\
    --batch_size ${batch_size}\
    --val_traj ${val_traj}\
    --ckpt_path ${ckpt_path}