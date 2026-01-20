from os.path import join
import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
from copy import deepcopy
from utils.utils import get_feat_types

class FeatEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.feat_types = get_feat_types(args)
        self.avg_feat = {}
        avg_feats = pickle.load(open(
            join('..', 'dataset', args.dataset, 'feature',
            'train_{}_{}_2stage_avg.pkl'.format(args.train_traj, args.clip_len)),"rb"))
        for type_ in self.feat_types:
            self.avg_feat[type_] = avg_feats[type_]
            
        # self.avg_feat['v3d_feat'] = self.avg_feat['v3d_feat'][:2048]
        # self.avg_feat['v2d_feat'] = self.avg_feat['v2d_feat'][:2048]
            
        if args.dataset == "vidor":
            self.staticEmb = nn.Sequential(
                nn.Linear(42 + 2048, args.clip_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.clip_hidden_dim, args.clip_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout))
                
            self.dynamicEmb = nn.Sequential(
                nn.Linear(42 + 2048, args.clip_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.clip_hidden_dim, args.clip_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout))

            self.lanEmb = nn.Sequential(
                nn.Linear(600, args.clip_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.clip_hidden_dim, args.clip_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout))
        else:
            self.staticEmb = nn.Sequential(
                nn.Linear(2048+42, args.clip_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout))
                
            self.dynamicEmb = nn.Sequential(
                nn.Linear(2048+42, args.clip_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout))

            self.lanEmb = nn.Sequential(
                nn.Linear(600, args.clip_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout))


    def forward(self, inputs, interv=None):
        
        if interv and not self.training:
            for type_ in self.feat_types:
                if interv[type_] == 'zero':
                    inputs[type_] = torch.zeros(*inputs[type_].shape).cuda()
                elif interv[type_] == 'avg':
                    inputs[type_][:, :] = torch.from_numpy(self.avg_feat[type_]).to(torch.float32).cuda()

        for type_ in self.feat_types:
            if "v2d" in type_ or "v3d" in type_:
                inputs[type_] = inputs[type_][:, :2048]
        
        outputs = {}
        outputs['static_feat'] = torch.Tensor().cuda()
        outputs['dynamic_feat'] = torch.Tensor().cuda()
        outputs['lan_feat'] = torch.Tensor().cuda()
        for type_ in self.feat_types: # only the choosed feature type willed be embedded
            if type_ in ['v2d_feat', 'rel_feat']:
                outputs['static_feat'] = torch.cat((outputs['static_feat'], inputs[type_]), -1)
            elif type_ in ['v3d_feat', 'mot_feat']:
                outputs['dynamic_feat'] = torch.cat((outputs['dynamic_feat'], inputs[type_]), -1)
            elif type_ in ['lan_feat']:
                outputs['lan_feat'] = torch.cat((outputs['lan_feat'], inputs[type_]), -1)

        outputs['static_feat'] = self.staticEmb(outputs['static_feat'])        
        outputs['dynamic_feat'] = self.dynamicEmb(outputs['dynamic_feat'])
        outputs['lan_feat'] = self.lanEmb(outputs['lan_feat'])
        
        return outputs
          
    
class Model(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        ds2dim = {"vidor":50, "vidvrd":132}
        self.clip_pred_dim = ds2dim[args.dataset]
        self.feat_types = get_feat_types(args)
        self.featEmbedding = FeatEmbedding(args)

        self.staticPred = nn.Linear(args.clip_emb_dim, self.clip_pred_dim)
        self.dynamicPred = nn.Linear(args.clip_emb_dim, self.clip_pred_dim)
        self.lanPred = nn.Linear(args.clip_emb_dim, self.clip_pred_dim)

    def _cal_logits(self, inputs):

        outputs = []
        for type_ in inputs:
            type2pred = getattr(self, type_[:-5] + "Pred")
            outputs.append(type2pred(inputs[type_]))
        outputs = sum(outputs)

        return outputs

    def forward(self, inputs, interv=None):

        for k in inputs:
            inputs[k] = inputs[k].cuda()
        
        outputs = self.featEmbedding(deepcopy(inputs))
        outputs = self._cal_logits(outputs)

        # if interv and not self.training:
        #     interv_outputs = self.featEmbedding(deepcopy(inputs), interv)
        #     interv_outputs = self._cal_logits(interv_outputs)
        #     outputs = torch.softmax(outputs,dim=-1) - torch.softmax(interv_outputs,dim=-1)
        #     # outputs = torch.softmax(outputs - interv_outputs,dim=-1)
        # elif not self.training:
        #     outputs = torch.softmax(outputs,dim=-1)

        if interv and not self.training:
            interv_outputs = self.featEmbedding(deepcopy(inputs), interv)
            interv_outputs = self._cal_logits(interv_outputs)
            outputs = torch.sigmoid(outputs) - torch.sigmoid(interv_outputs)
        elif not self.training:
            outputs = torch.sigmoid(outputs)

        return outputs
            
