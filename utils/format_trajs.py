import os
from os.path import join
import json
from collections import defaultdict

import numpy as np

from utils.parser_func import parse_args

def format_gt_trajs(data):
    tid2class = {}
    for obj in data["subject/objects"]:
        tid2class[obj['tid']] = object2id[obj['category']]
    trajs = data["trajectories"]

    traj_queues = defaultdict(dict)
    traj_list = []
    for frame_id, frame_objs in enumerate(trajs):
        unconnected_tids = list(traj_queues.keys())
        for obj in frame_objs: 
            if obj["tid"] in unconnected_tids:
                unconnected_tids.remove(obj["tid"])
            else:
                traj_queues[obj["tid"]]["traj"] = []
                traj_queues[obj["tid"]]["begin_fid"] = frame_id
            traj_queues[obj["tid"]]["traj"].append([obj["bbox"]["xmin"],obj["bbox"]["ymin"],obj["bbox"]["xmax"],obj["bbox"]["ymax"]])
        for tid in unconnected_tids:
            traj_queues[tid]["end_fid"] = frame_id
            traj_list.append({
                "class":tid2class[tid], 
                "score":1.0, 
                "traj":traj_queues[tid]["traj"],
                "begin_fid":traj_queues[tid]["begin_fid"],
                "end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)
    tids = list(traj_queues.keys())
    for tid in tids:
            traj_queues[tid]["end_fid"] = len(trajs)
            traj_list.append({
                "class":tid2class[tid], 
                "score":1.0, 
                "traj":traj_queues[tid]["traj"],
                "begin_fid":traj_queues[tid]["begin_fid"],
                "end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)

    return traj_list

def format_vru21_trajs(data):
    



def gen_pairs(trajs):
    traj_queues = defaultdict(dict)
    traj_list = []
    for frame_id, frame_objs in enumerate(trajs):
        unconnected_tids = list(traj_queues.keys())
        for obj in frame_objs: 
            if obj["tid"] in unconnected_tids:
                unconnected_tids.remove(obj["tid"])
            else:
                traj_queues[obj["tid"]]["traj"] = []
                traj_queues[obj["tid"]]["begin_fid"] = frame_id
            traj_queues[obj["tid"]]["traj"].append([obj["bbox"]["xmin"],obj["bbox"]["ymin"],obj["bbox"]["xmax"],obj["bbox"]["ymax"]])
        for tid in unconnected_tids:
            traj_queues[tid]["end_fid"] = frame_id
            traj_list.append({"tid":tid,"traj":traj_queues[tid]["traj"],"begin_fid":traj_queues[tid]["begin_fid"],"end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)
    tids = list(traj_queues.keys())
    for tid in tids:
            traj_queues[tid]["end_fid"] = len(trajs)
            traj_list.append({"tid":tid,"traj":traj_queues[tid]["traj"],"begin_fid":traj_queues[tid]["begin_fid"],"end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)

    trajs = traj_list
    pairs = []
    for sbj in trajs:
        for obj in trajs:
            if sbj["tid"] == obj["tid"]:
                continue
            if sbj["end_fid"] < obj["begin_fid"] or obj["end_fid"] < sbj["begin_fid"]:
                continue
            begin_fid = max(sbj["begin_fid"], obj["begin_fid"])
            end_fid = min(sbj["end_fid"], obj["end_fid"])
            if end_fid - begin_fid < 10:
                continue
            pairs.append({
                "sbj_tid":sbj["tid"],
                "sbj_traj":[sbj["traj"][i] for i in range(begin_fid-sbj["begin_fid"],end_fid-sbj["begin_fid"])],
                "obj_tid":obj["tid"],
                "obj_traj":[obj["traj"][i] for i in range(begin_fid-obj["begin_fid"],end_fid-obj["begin_fid"])],
                "begin_fid":begin_fid,
                "end_fid":end_fid
                })

    return pairs
    





