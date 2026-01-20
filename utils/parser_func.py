import argparse
import ast
from math import sqrt

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Build a video relation detection network')

    parser.add_argument('--dataset', dest='dataset',
                        help='The name of dataset to be used',
                        type=str, default='vidor')
    parser.add_argument('--clip_len', dest='clip_len',
                        help='Atomatic clip length in training and test',
                        type=int, default=30)
    parser.add_argument('--use_unlabeld_pair', dest='use_unlabeld_pair',
                        help='Whether to use unlabeled trajectory pairs for training',
                        action='store_true')                                                    
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size in training',
                        type=int, default=32)
    parser.add_argument('--lr', dest='lr',
                        help='Initial learning rate',
                        type=float, default=0.02)
    parser.add_argument('--dropout', dest='dropout',
                        help='Dropout value',
                        type=float, default=0.5)
    parser.add_argument('--num_layers', dest='num_layers',
                        help='Lstm Number of Layers',
                        type=int, default=2)
    parser.add_argument('--momentum', dest='momentum',
                        help='The momentum of SGD optimizer',
                        type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='Model weight decay value',
                        type=float, default=0.0001)
    parser.add_argument('--max_epoch', dest='max_epoch',
                        help='The epoch to stop',
                        type=int, default=40)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='The epoch to run from',
                        type=int, default=0)
    parser.add_argument('--resume', dest='resume',
                        help='Whether to resume the training',
                        action='store_true')
    parser.add_argument('--ckpt_path', dest='ckpt_path',
                        help='The checkpoint saved path',
                        type=str, default="")
    parser.add_argument('--print_freq', dest='print_freq',
                        help='The batch frequence of printing info',
                        type=int, default=100)
    parser.add_argument('--clip_top_n', dest='clip_top_n',
                        help='The top n predictions of clip to be saved',
                        type=int, default=20)
    parser.add_argument('--max_per_video', dest='max_per_video',
                        help='Max number of relations for each video to be saved',
                        type=int, default=200)
    parser.add_argument('--ps', dest='ps',
                        help='The P.S. information for this training process',
                        type=str, default="")
    parser.add_argument('--train_traj', dest='train_traj',
                        help='The trajectories for training split',
                        type=str, default="gt")
    parser.add_argument('--val_traj', dest='val_traj',
                        help='The trajectories source for validation split',
                        type=str, default="gt")
    parser.add_argument('--test_traj', dest='test_traj',
                        help='The trajectories source for testing split',
                        type=str, default="gt")
    parser.add_argument('--use_prior', dest='use_prior',
                        help='Wether to use prior or not',
                        action='store_true')
    parser.add_argument('--temp_model', dest='temp_model',
                        help='The temporal model used to encoding context',
                        default=None)

    parser.add_argument('--rel_feat', dest='rel_feat',
                        help='Use relative location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--mask_feat', dest='mask_feat',
                        help='Use mask location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--lan_feat', dest='lan_feat',
                        help='Use language feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--v2d_feat', dest='v2d_feat',
                        help='Use visual 2d feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--mot_feat', dest='mot_feat',
                        help='Use motion location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--v3d_feat', dest='v3d_feat',
                        help='Use visual 3d feature or not',
                        type=ast.literal_eval, default=True)

    parser.add_argument('--rel_emb_dim', dest='rel_emb_dim',
                        help='The dimension of relative location feature',
                        type=int, default=256)
    parser.add_argument('--mask_emb_dim', dest='mask_emb_dim',
                        help='The dimension of mask location feature',
                        type=int, default=256)
    parser.add_argument('--lan_emb_dim', dest='lan_emb_dim',
                        help='The dimension of language feature',
                        type=int, default=256)
    parser.add_argument('--v2d_emb_dim', dest='v2d_emb_dim',
                        help='The dimension of visual 2d feature',
                        type=int, default=512)
    parser.add_argument('--mot_emb_dim', dest='mot_emb_dim',
                        help='The dimension of mition location feature',
                        type=int, default=256)
    parser.add_argument('--v3d_emb_dim', dest='v3d_emb_dim',
                        help='The dimension of visual 3d feature',
                        type=int, default=512)
    parser.add_argument('--clip_hidden_dim', dest='clip_hidden_dim',
                        help='The dimension of clip embedding hidden layer',
                        type=int, default=1024)
    parser.add_argument('--clip_emb_dim', dest='clip_emb_dim',
                        help='The dimension of clip embedding output layer',
                        type=int, default=512)
    parser.add_argument('--temp_hidden_dim', dest='temp_hidden_dim',
                        help='The dimension of temporal unit hidden layer',
                        type=int, default=512)                  
    parser.add_argument('--layer', dest='layer',
                        help='The number of SGA layer',
                        type=int, default=1)         
    parser.add_argument('--clipheadnum', dest='clipheadnum',
                        help='The number of clip head',
                        type=int, default=1)        
    parser.add_argument('--videoheadnum', dest='videoheadnum',
                        help='The number of video head',
                        type=int, default=1)
    parser.add_argument('--headnum', dest='headnum',
                        help='The number of head',
                        type=int, default=1)    
    parser.add_argument('--clip_feat_dim', dest='clip_feat_dim',
                        help='The dimension of clip feature',
                        type=int, default=50)     
    parser.add_argument('--model', dest='model',
                        type=str, default="ViT-L/14") 
    parser.add_argument('--n_ctx', dest='n_ctx',
                        type=int, default=16)    
    parser.add_argument('--class_token_position', dest='class_token_position',
                        type=str, default="end")   
    parser.add_argument('--csc', dest='csc',
                        help='The epoch to run from',
                        type=bool, default=True)
    parser.add_argument('--k', dest='k',
                        help='The number of atom relationship',
                        type=int, default=160)
    
    args = parser.parse_args()
    return args 