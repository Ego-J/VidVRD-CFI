import numpy as np
from collections import defaultdict
from os.path import join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import visdom

from dataset import Dataset, padding_collate_fn
from utils.parser_func import parse_args
from utils.video_relation_detection import evaluate
from utils.utils import BCEFocalLoss, CEFocalLoss, get_feat_types, AverageMeter, get_logger, print_results
from utils.post_process import process_pred, association, format_
from model import Model

if __name__ == '__main__':
    
    args = parse_args()

    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = \
        args.dataset+ \
        "_clen"+str(args.clip_len)+ \
        "_lr"+str(args.lr)+ \
        "_bs"+str(args.batch_size)+ \
        feat_config+args.ps
    vis = visdom.Visdom(env=env_config)

    logger = get_logger(join('.','log',env_config+'_train.log'))
    logger.info('Experiment Config: {}'.format(args))

    logger.info("Preparing data from %s..."%args.dataset)
    train_dataset = Dataset(args, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=padding_collate_fn)
    if args.dataset == "vidor":
        val_dataset = Dataset(args, "val")
    elif args.dataset == "vidvrd":
        val_dataset = Dataset(args, "test")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn)

    model = Model(args).cuda()
    if args.resume:
        ckpt = torch.load(args.ckpt_path)
        args.start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = CEFocalLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=-1)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    lr_lambda = lambda epoch: (0.1**(epoch//10 + 1))*(10 - epoch%10) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    best_mean_ap = 0

    epoch_loss = AverageMeter()
    for epoch in range(args.start_epoch+1, args.max_epoch+1):
        logger.info("Training data from %s..."%args.dataset)
        model.train()
        batch_loss = AverageMeter()
        for idx, data in enumerate(tqdm(train_loader)):
            labels = data['label'].type(torch.long).view(-1, 1)
            labels = torch.zeros(labels.shape[0], 50).scatter_(1, labels, 1).cuda()
            feats = {}
            for k in data:
                if k != 'label':
                    feats[k] = data[k]
            preds = model(feats)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

            batch_loss.update(loss.item()/args.batch_size, args.batch_size)
            epoch_loss.update(loss.item()/args.batch_size, args.batch_size)
        
            if (idx+1) % args.print_freq == 0:
                vis.line([batch_loss.avg], [(epoch-1)*len(train_loader) + idx+1], win="Batch Loss", update='append', opts=dict(title='batch_loss'))
                batch_loss.reset()

        vis.line([epoch_loss.avg], [epoch], win="Train Epoch Loss", update='append', opts=dict(title='epoch_loss'))
        logger.info('Epoch: [{0}] \t LR: [{1}] \t Avg Train Loss:  {loss.avg:.4f}'.\
            format(epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss=epoch_loss))

        logger.info("Evaluating data from %s..."%args.dataset)        
        model.eval()
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
        results = evaluate(val_dataset.gt_rels, pred_rels, val_dataset.id2pre)
        vis.line([results["mean_ap"]], [epoch], win="Detection mAP", update='append', opts=dict(title='mean_ap'))
        vis.line([[results["rec_at_n"][50], results["rec_at_n"][100]]], [epoch], win="Detection Recall", update='append', opts=dict(title='recall', legend=['r@50', 'r@100']))
        vis.line([[results["mprec_at_n"][1], results["mprec_at_n"][5], results["mprec_at_n"][10]]], [epoch], win="Tagging Precision", update='append', opts=dict(title='precison', legend=['p@1', 'p@5', 'p@10']))
        print_results(logger, results)

        if results["mean_ap"] > best_mean_ap:
            state = {
                'epoch': epoch,
                'metric': {
                    'mean_ap': results["mean_ap"],
                    'rec_at_n': results["rec_at_n"],
                    'mprec_at_n': results["mprec_at_n"]},
                'config': args,
                'state_dict': model.state_dict()}
            ckpt_path = join('..','dataset',args.dataset,'model',env_config+".pth")
            torch.save(state, ckpt_path)
            best_mean_ap = results["mean_ap"]

        epoch_loss.reset()
        scheduler.step()
        vis.save([env_config])