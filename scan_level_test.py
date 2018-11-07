#-*-encoding:utf-8-*-
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from loss import FocalLoss
from retinanet3d import RetinaNet
from data_generator import SPHDataset
import config
from encoder import DataEncoder
from utils import box_iou, change_box_order, box_nms, box_distance
import math
import numpy as np
import time

def calc_scan_coord(boxes, start_coord):
    '''
    Calculate locations in scans

    Args:
      boxes: (FloatTensor) object locations in cubes [N, 6] zyxzyx
      start_coord: (FloatTensor) cube start location in scans [3]
    
    Returns:
      scan_loc: (FloatTensor) object locations in scans [N, 6] zyxzyx
    '''
    boxes = change_box_order(boxes, order="zyxzyx2zyxdhw")
    loc_zyx = boxes[:, :3]
    loc_dhw = boxes[:, 3:]
    cube_loc = start_coord.unsqueeze(0).expand_as(loc_zyx)
    loc_zyx += cube_loc
    scan_locs = torch.cat([loc_zyx, loc_dhw], 1)
    scan_locs = change_box_order(scan_locs, order="zyxdhw2zyxzyx")

    return scan_locs

def iter_scan(scan, scan_array, patient_df, net, cube_size=64, stride=50, iou=0.01):
    scan_df = pd.DataFrame(columns=["scan_id", "z", "y", "x", "iou"])
    start_time = time.time()
    gt_boxes, gt_labels = annotation(patient_df)
    #print(gt_boxes, gt_labels)
    ais_gt_boxes, mia_gt_boxes = split_class(gt_boxes, gt_labels)
    #print(ais_gt_boxes, mia_gt_boxes)
    ais_locs = torch.FloatTensor(1,6)
    ais_probs = torch.FloatTensor(1)

    mia_locs = torch.FloatTensor(1,6)
    mia_probs = torch.FloatTensor(1)

    for z in range(0, scan_array.shape[0], stride):
        for y in range(0, scan_array.shape[1], stride):
            for x in range(0, scan_array.shape[2], stride):
                start_coord = torch.FloatTensor([z,y,x])
                end_coord = start_coord + torch.FloatTensor([cube_size, cube_size, cube_size])
                zmax = min(z+cube_size, scan_array.shape[0])
                ymax = min(y+cube_size, scan_array.shape[1])
                xmax = min(x+cube_size, scan_array.shape[2])
                cube_sample = np.zeros((cube_size,cube_size,cube_size), dtype=np.float32)
                cube_sample[:(zmax-z), :(ymax-y),:(xmax-x)] = scan_array[z:zmax, y:ymax, x:xmax]
                cube_sample = np.expand_dims(cube_sample, 0)
                cube_sample = np.expand_dims(cube_sample, 0)
                input_cube = Variable(torch.from_numpy(cube_sample).cuda())
                locs, clss = net(input_cube)
                locs = locs.data.cpu().squeeze()
                clss = clss.data.cpu().squeeze()
                ais_boxes, ais_scores, ais_labels, mia_boxes, mia_scores, mia_labels = DataEncoder().decode(locs, clss, 
                                                                                        [cube_size, cube_size, cube_size])
                if not isinstance(ais_boxes, int):
                    ais_boxes = calc_scan_coord(ais_boxes, start_coord)
                    ais_locs = torch.cat([ais_locs, ais_boxes], 0)
                    ais_probs = torch.cat([ais_probs, ais_scores], 0)

                if not isinstance(mia_boxes, int):
                    mia_boxes = calc_scan_coord(mia_boxes, start_coord)
                    mia_locs = torch.cat([mia_locs, mia_boxes], 0)
                    mia_probs = torch.cat([mia_probs, mia_scores], 0)

    end_time = time.time()
    run_time = end_time - start_time
    print(run_time) 
    if not isinstance(ais_gt_boxes, int):
        ais_locs = ais_locs[1:, :]
        ais_probs = ais_probs[1:]
        ais_keep = box_nms(ais_locs, ais_probs)
        ais_locs = ais_locs[ais_keep]
        ais_probs = ais_probs[ais_keep]
        ais_count, best_ious = find_best_pred(ais_gt_boxes, ais_locs)
        ais_locs = change_box_order(ais_locs, "zyxzyx2zyxdhw")
        for i in range(ais_locs.size(0)):
            insert = {"scan_id":scan, "z":ais_locs[i,0], "y":ais_locs[i,1], "x":ais_locs[i,2], "iou":best_ious[i]}
            la_df = pd.DataFrame(data=insert, index=["0"])
            scan_df = scan_df.append(la_df, ignore_index=True)

    else:
        ais_count = np.zeros(3)
    
    if not isinstance(mia_gt_boxes, int):
        mia_locs = mia_locs[1:, :]
        mia_probs = mia_probs[1:]
        mia_keep = box_nms(mia_locs, mia_probs)
        mia_locs = mia_locs[mia_keep]
        mia_probs = mia_probs[mia_keep]
        mia_count, best_ious = find_best_pred(mia_gt_boxes, mia_locs)
        for i in range(mia_locs.size(0)):
            insert = {"scan_id":scan, "z":mia_locs[i,0], "y":mia_locs[i,1], "x":mia_locs[i,2], "iou":best_ious[i]}
            la_df = pd.DataFrame(data=insert, index=["0"])
            scan_df = scan_df.append(la_df, ignore_index=True)
    else:
        mia_count = np.zeros(3)
    
    return ais_count, mia_count, scan_df


def annotation(patient_df):
    '''
    Extract bounding box labels from dataframe

    Args:
      patient_df: (pandas DataFrame) patient dataframe 
    
    Returns:
      gt_boxes: (tensor) [N, 6] zyxzyx
      gt_labels: (LongTensor) [N, ]
    '''
    #print(patient_df)
    gt_boxes = []
    gt_labels = []

    for i in range(patient_df.shape[0]):
        zmin = patient_df.iloc[i, 6]
        ymin = patient_df.iloc[i, 7]
        xmin = patient_df.iloc[i, 8]
        zmax = patient_df.iloc[i, 9]
        ymax = patient_df.iloc[i, 10]
        xmax = patient_df.iloc[i, 11]
        if patient_df.iloc[i, 2]=="原位腺癌" or patient_df.iloc[i, 2]=="不典型腺瘤样增生":
            label = 1
        else:
            label = 2
        box = [zmin, ymin, xmin, zmax, ymax, xmax]
        gt_boxes.append(box)
        gt_labels.append(label)
    #print(gt_boxes, gt_labels)
    gt_boxes = torch.FloatTensor(gt_boxes)
    gt_labels = torch.LongTensor(gt_labels)
    #print(gt_boxes, gt_labels)
    #gt_ids = torch.LongTensor(gt_ids)

    return gt_boxes, gt_labels


def split_class(boxes, labels):
    ais_ids = (labels==1)
    ais_num = ais_ids.long().sum()
    if ais_num == 0:
        ais_box = 0

    else:
        ais_ids = ais_ids.nonzero().squeeze()
        ais_box = boxes[ais_ids]

    mia_ids = (labels==2)
    mia_num = mia_ids.long().sum()
    if mia_num == 0:
        mia_box = 0

    else:
        mia_ids = mia_ids.nonzero().squeeze()
        mia_box = boxes[mia_ids]
    
    return ais_box, mia_box


def find_best_pred(gt_boxes, pred_boxes):
    '''
    Find whether there is a predicted box for each ground box

    Args:
      gt_boxes: (FloatTensor) [N, 6]  zyxzyx
      pred_boxes: (FloatTensor) [M, 6]   zyxzyx
    
    Returns:
      count: (ndarray) (tp, fn, fp)
    '''
    tp = 0
    fn = 0
    fp = 0
    distance = box_distance(gt_boxes, pred_boxes)
    iou = box_iou(gt_boxes, pred_boxes)
    min_dists, min_ids = distance.min(1)
    best_ious, best_ids = iou.min(0)   # find best gt for predict
    gt_boxes = change_box_order(gt_boxes, order="zyxzyx2zyxdhw")
    for i in range(gt_boxes.size(0)):
        gt = gt_boxes[i,:]
        diameter = math.sqrt(gt[3]**2 + gt[4]**2 + gt[5]**2)
        radius = diameter/2 + 10.
        if min_dists[i] <= radius:
            tp += 1
        else:
            fn += 1
    fp = pred_boxes.size(0) - tp

    return np.array([tp, fn, fp]), best_ious


if __name__ == "__main__":
    import config
    import os

    cfg = config.Config()
    net = RetinaNet(backbone="resnet34", num_classes=2)
    checkpoint = torch.load(cfg.checkpoints_path+"ckpts_v19_resnet34_gamma0_change_ratio_train/"+"ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    net.cuda()
    net.eval()
    label_df = pd.read_csv(cfg.test_label)
    scan_list = label_df["patient_ID"].drop_duplicates().tolist()
    
    result_df = pd.DataFrame(columns=["scan_id", "z", "y", "x", "iou"])
    
    all_ais = np.zeros(3)
    all_mia = np.zeros(3)
    for scan in scan_list:
        print(scan)
        scan_name = os.path.join(cfg.test_path, scan+".npy")
        scan_array = np.load(scan_name)
        patient_df = label_df[label_df["patient_ID"] == scan]
        #start_time = time.time()
        ais_count, mia_count, scan_df = iter_scan(scan, scan_array, patient_df, net)
        #end_time = time.time()
        #run_time = end_time - start_time
        #print(run_time)
        print(ais_count, mia_count)
        all_ais += ais_count
        all_mia += mia_count
        result_df = pd.concat([result_df, scan_df], ignore_index=True)
        result_df.to_csv("/home/youkun/sph_samples/results/result_dist.csv", index=False) 
    print(all_ais, all_mia)
    '''
    scan_name = os.path.join(cfg.test_path, "V15189761.npy")
    df = label_df[label_df["patient_ID"]=="V15189761"]
    print(df)
    scan_array = np.load(scan_name) 
    ais_count, mia_count = iter_scan(scan_array, df, net)
    print(ais_count, mia_count)
    '''
