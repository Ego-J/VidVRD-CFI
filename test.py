from collections import defaultdict
from os.path import join
import copy
import json
from math import log

import torch
from torch.nn.modules.activation import Sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset, padding_collate_fn
from utils.parser_func import parse_args
from utils.video_relation_detection import evaluate
from utils.utils import get_feat_types, AverageMeter, get_logger, relation_filter, cal_entropy, print_results
from utils.post_process import process_pred, association, format_
from model import Model

if __name__ == '__main__':
    
    test_args = parse_args()
    ckpt = torch.load(test_args.ckpt_path)

    print(ckpt['metric'])
    args = ckpt['config']
    args.val_traj = test_args.val_traj
    args.test_traj = test_args.test_traj
    args.batch_size = test_args.batch_size

    if args.dataset == "vidor":
        val_dataset = Dataset(args, "val")
        traj_source = args.val_traj
    elif args.dataset == "vidvrd":
        val_dataset = Dataset(args, "test")
        traj_source = args.test_traj
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn)
    split = json.load(open(join('..', 'dataset', args.dataset, 'data', 'predicate_split.json'), 'r'))

    model = Model(args).cuda()
    model.load_state_dict(ckpt['state_dict'])
    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = \
        args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_drop"+str(args.dropout)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        "_"+traj_source+ \
        feat_config+args.ps
 
    logger = get_logger(join('.','log',env_config+'_test_ab.log'))
    logger.info('Experiment Config: {}'.format(args))

    softmax = torch.nn.Softmax(dim=-1)
    sigmoid = torch.nn.Sigmoid()
    
    model.eval()

    interv_types = {
                    "none":None,
                    "zero_static":{
                        'rel_feat': "zero",
                        'v2d_feat': "zero",
                        'lan_feat': None,
                        'mot_feat': None,
                        'v3d_feat': None
                    },
                    "avg_static":{
                        'rel_feat': "avg",
                        'v2d_feat': "avg",
                        'lan_feat': None,
                        'mot_feat': None,
                        'v3d_feat': None
                    },
                    "zero_dynamic":{
                        'rel_feat': None,
                        'v2d_feat': None,
                        'lan_feat': None,
                        'mot_feat': "zero",
                        'v3d_feat': "zero"
                    },
                    "avg_dynamic":{
                        'rel_feat': None,                       
                        'v2d_feat': None,
                        'lan_feat': None,
                        'mot_feat': "avg",
                        'v3d_feat': "avg"
                    },
                    "zero_lan":{
                        'rel_feat': None,
                        'v2d_feat': None,
                        'lan_feat': "zero",
                        'mot_feat': None,
                        'v3d_feat': None
                    },
                    "avg_lan":{
                        'rel_feat': None,
                        'v2d_feat': None,
                        'lan_feat': "avg",
                        'mot_feat': None,
                        'v3d_feat': None
                    }
                }

    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(val_loader):
            vids = data['vid']
            pair_data = data['pair_data']
            feats = {}
            for k in data:
                if k not in ['vid', 'pair_data']:
                    feats[k] = data[k]
            preds = model(feats)
            for pair_id, vid in enumerate(vids):
                rels = process_pred(args, val_dataset.id2pre, val_dataset.obj2id, val_dataset.prior, preds[pair_id], pair_data[pair_id])
                pred_rels[vid].extend(rels)
    for vid in pred_rels:
        pred_rels[vid] = format_(args, pred_rels[vid])
    results = evaluate(val_dataset.gt_rels, pred_rels, split['full'])
    logger.info('---------- Intervention Type: {} ----------'.format('none'))
    print_results(logger, results, verbose=True)  

    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(val_loader):
            vids = data['vid']
            pair_data = data['pair_data']
            feats = {}
            for k in data:
                if k not in ['vid', 'pair_data']:
                    feats[k] = data[k]
            full_preds = model(copy.deepcopy(feats), interv = interv_types["none"])        
            static_preds = model(copy.deepcopy(feats), interv = interv_types["zero_static"])
            dynamic_preds = model(copy.deepcopy(feats), interv = interv_types["zero_dynamic"])
            lan_preds = model(copy.deepcopy(feats), interv = interv_types["zero_lan"])

            full_ent = cal_entropy(softmax(full_preds)).view(-1,1)
            static_ent = cal_entropy(softmax(static_preds)).view(-1,1)
            dynamic_ent = cal_entropy(softmax(dynamic_preds)).view(-1,1)
            lan_ent = cal_entropy(softmax(lan_preds)).view(-1,1)
            ent = torch.cat((full_ent, static_ent, dynamic_ent, lan_ent), dim=-1)

            score = softmax(ent)
            full_score = score[:, 0:1]
            static_score = score[:, 1:2]
            dynamic_score = score[:, 2:3]
            lan_score = score[:, 3:]

            preds = full_preds*full_score + static_preds*static_score +\
                    dynamic_preds*dynamic_score + lan_preds*lan_score
            
            for pair_id, vid in enumerate(vids):
                rels = process_pred(args, val_dataset.id2pre, val_dataset.obj2id, val_dataset.prior, preds[pair_id], pair_data[pair_id])
                pred_rels[vid].extend(rels)
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
    
    logger.info('---------- Intervention Type: {} ----------'.format("zero_3+1fusion_ent"))
    results = evaluate(val_dataset.gt_rels, pred_rels, split["full"])
    print_results(logger, results, verbose=True)


    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(val_loader):
            vids = data['vid']
            pair_data = data['pair_data']
            feats = {}
            for k in data:
                if k not in ['vid', 'pair_data']:
                    feats[k] = data[k]
            full_preds = model(copy.deepcopy(feats), interv = interv_types["none"])        
            static_preds = model(copy.deepcopy(feats), interv = interv_types["avg_static"])
            dynamic_preds = model(copy.deepcopy(feats), interv = interv_types["avg_dynamic"])
            lan_preds = model(copy.deepcopy(feats), interv = interv_types["avg_lan"])
            
            full_ent = cal_entropy(softmax(full_preds)).view(-1,1)
            static_ent = cal_entropy(softmax(static_preds)).view(-1,1)
            dynamic_ent = cal_entropy(softmax(dynamic_preds)).view(-1,1)
            lan_ent = cal_entropy(softmax(lan_preds)).view(-1,1)
            ent = torch.cat((full_ent, static_ent, dynamic_ent, lan_ent), dim=-1)

            score = softmax(ent)
            full_score = score[:, 0:1]
            static_score = score[:, 1:2]
            dynamic_score = score[:, 2:3]
            lan_score = score[:, 3:]

            preds = full_preds*full_score + static_preds*static_score +\
                    dynamic_preds*dynamic_score + lan_preds*lan_score

            for pair_id, vid in enumerate(vids):
                rels = process_pred(args, val_dataset.id2pre, val_dataset.obj2id, val_dataset.prior, preds[pair_id], pair_data[pair_id])
                pred_rels[vid].extend(rels)
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
    logger.info('---------- Intervention Type: {} ----------'.format("avg_3+1fusion_ent"))
    results = evaluate(val_dataset.gt_rels, pred_rels, split["full"])
    print_results(logger, results, verbose=True)