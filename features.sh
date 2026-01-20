#!/bin/sh
gpu_id=0

clip_len=30
rel_feat=True
mask_feat=False
lan_feat=True
v2d_feat=True
mot_feat=True
v3d_feat=True
dataset=vidor

CUDA_VISIBLE_DEVICES=$gpu_id python features.py\
	--clip_len ${clip_len}\
    --rel_feat ${rel_feat}\
    --mask_feat ${mask_feat}\
    --lan_feat ${lan_feat}\
    --v2d_feat ${v2d_feat}\
    --mot_feat ${mot_feat}\
    --v3d_feat ${v3d_feat}\
    --dataset ${dataset}\
    --use_unlabeld_pair
    #--use_gt_traj

 


