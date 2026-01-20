import os
import numpy as np
import pickle
import json
from collections import defaultdict
from os.path import join

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):

    def __init__(self, args, split):
        super().__init__()
        self.dataset = args.dataset
        self.split = split
        if split=='train':
            feat_path = 'train_{}_{}'.format(args.train_traj, args.clip_len)  
        elif split=='val':
            self.gt_rels = json.load(open(
                join('..', 'dataset', self.dataset, 'data', 'val_relation_gt.json'),"r"))
            feat_path = 'val_{}_{}'.format(args.val_traj, args.clip_len)
        elif split=='test':
            self.gt_rels = json.load(open(
                join('..', 'dataset', self.dataset, 'data', 'test_relation_gt.json'),"r"))
            feat_path = 'test_{}_{}'.format(args.test_traj, args.clip_len)
        
        # feat_path += '_2stage'

        self.FEAT_ROOT = join('..', 'dataset', self.dataset, 'feature', feat_path)
        self.path_list = os.listdir(self.FEAT_ROOT)
        self.id2pre = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'id2predicate.json'),"r"))
        self.pre2id = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'predicate2id.json'),"r"))
        self.id2obj = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'id2object.json'),"r"))
        self.obj2id = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'object2id.json'),"r"))
        self.pre_num = len(self.id2pre)
        self.prior = pickle.load(open(join('..', 'dataset', self.dataset, 'data', 'prior.pkl'),'rb'))

    def __getitem__(self, index):

        pair_path = self.path_list[index]
        pair_data = pickle.load(open(join(self.FEAT_ROOT, pair_path),"rb"))
        item = {}
        for type_ in pair_data[0]:
            item[type_] = np.array(pair_data[0][type_])
        
        if self.split == 'train':
            # reorganize the pair label into numpy array
            pair_label = np.zeros((0,self.pre_num),)
            for clip_label in pair_data[1]:
                tmp_label = np.zeros(self.pre_num,)
                if len(clip_label) > 0:
                    tmp_label[clip_label] = 1
                pair_label = np.vstack((pair_label,tmp_label))
            item['label'] = pair_label
            # reorganize the pair feats into numpy array
        else:
            item['vid'] = pair_path.split('.')[0][:-7]
            item['pair_data'] = pair_data[1]
        return item

    # def __getitem__(self, index):

    #     pair_path = self.path_list[index]
    #     pair_data = pickle.load(open(join(self.FEAT_ROOT, pair_path),"rb"))
    #     item = {}
    #     for type_ in pair_data[0]:
    #         item[type_] = np.array(pair_data[0][type_])
        
    #     if self.split == 'train':
    #         item['label'] = pair_data[1]
    #     else:
    #         item['vid'] = pair_path.split('.')[0][:-7]
    #         item['pair_data'] = pair_data[1]
    #     return item

    def __len__(self):
        return len(self.path_list)

def padding_collate_fn(batch):
    seq_lens = torch.LongTensor([len(x['rel_feat']) for x in batch])
    batch_data = {}
    for k in batch[0]:
        if k == 'mask_feat': continue
        batch_data[k] = [x[k] for x in batch]
        if k in ['vid', 'pair_data']:
            continue
        else:    
            batch_data[k] = pad_sequence([torch.from_numpy(x).type(torch.float32) for x in batch_data[k]], batch_first=True)
    return batch_data, seq_lens