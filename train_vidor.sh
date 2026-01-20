#!/bin/sh
gpu_id=1
lr=0.01
max_epoch=40
batch_size=100
clip_len=30
ckpt='your/ckpt/path'
ps='2stage_3indep_bce'
rel_feat=True
mask_feat=False
lan_feat=True
v2d_feat=True
mot_feat=True
v3d_feat=True
val_traj=gt
dataset=vidor

CUDA_VISIBLE_DEVICES=$gpu_id python train.py\
	--lr ${lr} \
	--batch_size ${batch_size}\
	--ps ${ps}\
	--clip_len ${clip_len}\
	--max_epoch ${max_epoch}\
	--rel_feat ${rel_feat}\
    --mask_feat ${mask_feat}\
    --lan_feat ${lan_feat}\
    --v2d_feat ${v2d_feat}\
    --mot_feat ${mot_feat}\
    --v3d_feat ${v3d_feat}\
	--val_traj ${val_traj}\
	--dataset ${dataset}\
	# --ckpt ${ckpt}\
	# --resume
