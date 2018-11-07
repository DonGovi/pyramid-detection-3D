import argparse
import pandas as pd
import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
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
from utils import change_box_order, box_nms, box_distance

def calc_scan_coord(boxes, start_coord):
    loc_zyx = boxes[:, :3]
    loc_dhw = boxes[:, 3:]
    cube_loc = start_coord.unsqueeze(0).expand_as(loc_zyx)
    loc_zyx += cube_loc
    scan_loc = torch.cat([loc_zyx, loc_dhw], 1)

    return scan_loc

def iter_scan(scan_file, net, cube_size=64, stride=50, threshold=0.01):
    scan_array = np.load(scan_file)
    '''
    scan_array = (scan_array + 1000.)/(400. + 1000.)
    scan_array = np.clip(scan_array, 0, 1)
    '''
    ais_locs = torch.FloatTensor(1,6)
    ais_probs = torch.FloatTensor(1)
    ais_classes = torch.LongTensor(1)
    mia_locs = torch.FloatTensor(1,6)
    mia_probs = torch.FloatTensor(1)
    mia_classes = torch.LongTensor(1)
    for z in range(0, scan_array.shape[0], stride):
        for y in range(0, scan_array.shape[1], stride):
            for x in range(0, scan_array.shape[2], stride):
                print(z,y,x)
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
                ais_boxes, ais_scores, ais_labels, mia_boxes, mia_scores, mia_labels = DataEncoder().decode(locs, clss, [cube_size, cube_size, cube_size])
                cube_loc = torch.FloatTensor([z,y,x])
    
                if not isinstance(ais_boxes, int):
                    ais_boxes = calc_scan_coord(ais_boxes, cube_loc)
                    ais_locs = torch.cat([ais_locs, ais_boxes], 0)
                    ais_probs = torch.cat([ais_probs, ais_scores], 0)
                    ais_classes = torch.cat([ais_classes, ais_labels], 0)
                    

                if not isinstance(mia_boxes, int):
                    #print(mia_scores.size(), mia_labels.size())
                    mia_boxes = calc_scan_coord(mia_boxes, cube_loc)
                    mia_locs = torch.cat([mia_locs, mia_boxes], 0)
                    mia_probs = torch.cat([mia_probs, mia_scores], 0)
                    mia_classes = torch.cat([mia_classes, mia_labels], 0)
    '''
    ais_locs = ais_locs[1:,:]
    ais_probs = ais_probs[1:]
    ais_classes = ais_classes[1:]
    ais_locs = change_box_order(ais_locs, "zyxdhw2zyxzyx")
    ais_keep = box_nms(ais_locs, ais_probs, threshold=threshold)
    ais_locs = ais_locs[ais_keep]
    ais_probs = ais_probs[ais_keep]
    ais_classes = ais_classes[ais_keep]

    mia_locs = mia_locs[1:,:]
    mia_probs = mia_probs[1:]
    mia_classes = mia_classes[1:]
    #print(mia_classes)
    mia_locs = change_box_order(mia_locs, "zyxdhw2zyxzyx")
    mia_keep = box_nms(mia_locs, mia_probs, threshold=threshold)
    mia_locs = mia_locs[mia_keep]
    mia_probs = mia_probs[mia_keep]
    mia_classes = mia_classes[mia_keep]
    
    
    la_locs = torch.cat((ais_locs, mia_locs), 0)
    la_probs = torch.cat((ais_probs, mia_probs), 0)
    la_labels = torch.cat([ais_classes, mia_classes], 0)
    
    la_locs = la_locs.numpy()
    la_probs = la_probs.numpy()
    la_labels = la_labels.numpy()
    
    neg = np.where(la_locs<=0)[0]
    la_locs = np.delete(la_locs, neg, axis=0)
    la_probs = np.delete(la_probs, neg, axis=0)
    la_labels = np.delete(la_labels, neg, axis=0)
    '''

    return la_locs, la_probs, la_labels
    

if __name__ == '__main__':
    cfg = config.Config()
    net = RetinaNet(backbone="resnet34", num_classes=2)
    checkpoint = torch.load(os.path.join(cfg.checkpoints_path, "ckpts_v19_resnet34_gamma0_change_ratio_train/", "ckpt.pth"))
    net.load_state_dict(checkpoint["net"])
    net.cuda()
    net.eval()
    scan_path = "/home/youkun/sph_samples/test/"
    scan_file = os.path.join(scan_path, "P10377452.npy")
    la_locs, la_probs, la_labels = iter_scan(scan_file, net)
    df = pd.DataFrame(columns=["zmin","ymin","xmin","zmax","ymax", "xmax", "prob", "label"])
    for i in range(la_locs.shape[0]):
        if la_probs[i] > 0.5:
            df.loc[i, "zmin"] = la_locs[i, 0]
            df.loc[i, "ymin"] = la_locs[i, 1]
            df.loc[i, "xmin"] = la_locs[i, 2]
            df.loc[i, "zmax"] = la_locs[i, 3]
            df.loc[i, "ymax"] = la_locs[i, 4]
            df.loc[i, "xmax"] = la_locs[i, 5]
            df.loc[i, "prob"] = la_probs[i]
            df.loc[i, "label"] = la_labels[i]

    df.to_csv("/home/youkun/sph_samples/results/P10377452.csv", index=False)




